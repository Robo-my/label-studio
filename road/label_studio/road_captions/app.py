"""road-captions: writes the training sentence live, while the intern reviews a box.

Runs in its own container next to an unchanged Label Studio. nginx sends /road-api/* here.

POST /road-api/sentence  {"task_id", "decision", "lookalike", "words", "prefetch"}
  -> {"ok", "label", "sentence", "source", "message"}
  source: bedrock (model wrote it), template (fixed or built from words), intern (words used as typed),
          lookalike (fixed look-alike sentence), none (no sentence needed)
GET  /road-api/health
GET  /road-api/static/road-captions.js   the browser module that nginx adds to Label Studio pages

Run: uvicorn road_captions.app:create_app --factory --host 0.0.0.0 --port 8000

Every sentence passes the caption rules in reward.py before it goes back. A sentence made from the
intern's words must also keep every content word the intern typed.
"""
import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .ls_client import Denied, LabelStudio, login_headers
from .sentences import (DECISIONS, FALLBACK, LOOKALIKE_CAPTIONS, LOOKALIKES, caption_prompt, check, clean_reply,
                        clean_words, intern_sentence, resolve_label, rewrite_prompt, template_sentence,
                        words_problem)

STATIC_DIR = Path(__file__).resolve().parent / "static"
log = logging.getLogger("road-captions")


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    ls_url: str = "http://label-studio:8080"
    model_id: str = ""                 # empty: no model, template sentences only
    region: str = "ap-southeast-5"
    send_image: bool = False           # True: the crop goes to Bedrock (see README: data residency)
    rewrite_sends_image: bool = False  # True: the crop also goes with a rewrite from the intern's words
    timeout_s: float = 10.0
    max_tokens: int = 2048
    extra_fields: Optional[dict] = None
    s3_region: str = "ap-southeast-5"
    allowed_buckets: tuple = ()
    image_field: str = "crop"
    max_image_bytes: int = 8_000_000
    rate_per_min: int = 60
    log_path: str = "/var/log/road-captions/requests.jsonl"

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        extra = env.get("BEDROCK_EXTRA_FIELDS", "").strip()
        return cls(
            ls_url=env.get("LS_URL", cls.ls_url),
            model_id=env.get("BEDROCK_MODEL_ID", "").strip(),
            region=env.get("BEDROCK_REGION", cls.region),
            send_image=_bool(env.get("SEND_IMAGE", "false")),
            rewrite_sends_image=_bool(env.get("REWRITE_SENDS_IMAGE", "false")),
            timeout_s=float(env.get("BEDROCK_TIMEOUT_S", cls.timeout_s)),
            max_tokens=int(env.get("BEDROCK_MAX_TOKENS", cls.max_tokens)),
            extra_fields=json.loads(extra) if extra else None,
            s3_region=env.get("S3_REGION", cls.s3_region),
            allowed_buckets=tuple(b.strip() for b in env.get("ALLOWED_S3_BUCKETS", "").split(",") if b.strip()),
            image_field=env.get("IMAGE_FIELD", cls.image_field),
            max_image_bytes=int(env.get("MAX_IMAGE_BYTES", cls.max_image_bytes)),
            rate_per_min=int(env.get("RATE_PER_MIN", cls.rate_per_min)),
            log_path=env.get("LOG_PATH", cls.log_path),
        )


class SentenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: int = Field(ge=1)
    decision: str = Field(max_length=40)
    lookalike: Optional[str] = Field(default=None, max_length=40)
    words: str = Field(default="", max_length=300)
    prefetch: bool = False


@dataclass
class Memory:
    """In-memory state. One worker process, so plain dicts are enough."""
    cache: OrderedDict = field(default_factory=OrderedDict)  # (kind, task, label, words) -> answer
    inflight: dict = field(default_factory=dict)              # same key -> Future, so one model call per key
    users: dict = field(default_factory=dict)                 # login hash -> (user, expires)
    hits: dict = field(default_factory=lambda: defaultdict(deque))
    max_cache: int = 20000

    def remember(self, key, answer):
        self.cache[key] = answer
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)


def answer(ok, label, sentence="", source="none", message="", error=""):
    return {"ok": ok, "label": label, "sentence": sentence, "source": source, "message": message, "error": error}


def create_app(settings=None, writer=None, ls_transport=None, s3_reader=None):
    settings = settings or Settings.from_env()
    if writer is None and settings.model_id:
        from .writer import BedrockWriter
        writer = BedrockWriter(settings.model_id, settings.region, settings.timeout_s, settings.max_tokens,
                               settings.extra_fields)
    if s3_reader is None and settings.allowed_buckets:
        from .writer import S3Reader
        s3_reader = S3Reader(settings.s3_region, settings.allowed_buckets)
    ls = LabelStudio(settings.ls_url, transport=ls_transport)
    mem = Memory()
    log_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await ls.close()

    app = FastAPI(title="road-captions", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/road-api/static", StaticFiles(directory=STATIC_DIR), name="static")

    def write_log(row):
        line = json.dumps(row, ensure_ascii=False)
        try:
            Path(settings.log_path).parent.mkdir(parents=True, exist_ok=True)
            with log_lock, open(settings.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            log.warning("cannot write %s; request: %s", settings.log_path, line)

    async def current_user(headers):
        key = hashlib.sha256(json.dumps(headers, sort_keys=True).encode()).hexdigest()
        hit = mem.users.get(key)
        if hit and hit[1] > time.monotonic():
            return hit[0]
        user = await ls.whoami(headers)
        if user is not None:
            mem.users[key] = (user, time.monotonic() + 60)
        return user

    def allowed(who):
        now, hits = time.monotonic(), mem.hits[who]
        while hits and hits[0] < now - 60:
            hits.popleft()
        if len(hits) >= settings.rate_per_min:
            return False
        hits.append(now)
        return True

    async def call_model(prompt, image=None, image_format=None):
        reply = await asyncio.wait_for(asyncio.to_thread(writer.write, prompt, image, image_format),
                                       timeout=settings.timeout_s + 5)
        return clean_reply(reply)

    async def once(key, make):
        """Cached answer, or one shared model call for all requests that need the same sentence."""
        if key in mem.cache:
            return {**mem.cache[key], "cached": True}
        if key in mem.inflight:
            return {**(await asyncio.shield(mem.inflight[key])), "cached": True}
        fut = asyncio.get_running_loop().create_future()
        mem.inflight[key] = fut
        try:
            result = await make()
        except Exception as exc:  # make() catches model errors itself; this is a last guard
            result = answer(False, None, message="The sentence service failed. Press the key again.",
                            error=f"{type(exc).__name__}: {exc}"[:300])
        except BaseException:  # cancelled: release the requests that wait for this call
            fut.cancel()
            raise
        finally:
            mem.inflight.pop(key, None)
        fut.set_result(result)
        if result["ok"] and result["source"] == "bedrock":
            mem.remember(key, result)
        return result

    async def from_image(label, data, headers, task_id):
        async def make():
            error = ""
            if writer is not None and settings.send_image:
                try:
                    image, fmt = await ls.image(data.get(settings.image_field), headers, settings.max_image_bytes,
                                                s3_reader)
                    sentence = await call_model(caption_prompt(label), image, fmt)
                    problem = check(sentence, label)
                    if not problem:
                        return answer(True, label, sentence, "bedrock")
                    error = f"model reply {problem}: {sentence[:120]!r}"
                except Exception as exc:  # any failure falls back below; the log keeps the reason
                    error = f"model call failed: {type(exc).__name__}: {exc}"[:300]
            elif writer is None:
                error = "no model configured"
            else:
                error = "SEND_IMAGE is off"
            if label == "no_defect":
                return answer(False, label, message="Type what it is in 2-6 words, then press Enter.", error=error)
            return answer(True, label, FALLBACK[label], "template",
                          message="Template sentence. To describe the box better, type 2-6 words and press Enter.",
                          error=error)
        return await once(("image", task_id, label), make)

    async def from_words(label, words, data, headers, task_id):
        problem = words_problem(label, words)
        if problem:
            return answer(False, label, message=problem)
        typed = intern_sentence(label, words)
        if typed:
            return answer(True, label, typed, "intern")

        async def make():
            notes = []
            if writer is not None:
                try:
                    image = fmt = None
                    if settings.send_image and settings.rewrite_sends_image:
                        image, fmt = await ls.image(data.get(settings.image_field), headers,
                                                    settings.max_image_bytes, s3_reader)
                    sentence = await call_model(rewrite_prompt(label, words), image, fmt)
                    problem = check(sentence, label, words)
                    if not problem:
                        return answer(True, label, sentence, "bedrock")
                    notes.append(f"model reply {problem}: {sentence[:120]!r}")
                except Exception as exc:
                    notes.append(f"model call failed: {type(exc).__name__}: {exc}"[:300])
            sentence = template_sentence(label, words)
            problem = check(sentence, label, words)
            if not problem:
                return answer(True, label, sentence, "template", error="; ".join(notes))
            notes.append(f"template {problem}: {sentence[:120]!r}")
            return answer(False, label, message="These words do not make a valid sentence. Change them, "
                                                "or press 7 (unsure).", error="; ".join(notes))
        return await once(("words", task_id, label, words.lower()), make)

    async def make_sentence(body, data, headers):
        if body.decision not in DECISIONS:
            return answer(False, None, message=f"Unknown key {body.decision!r}.")
        try:
            label, lookalike = resolve_label(body.decision, body.lookalike, data.get("detector_label", ""))
        except ValueError as exc:
            return answer(False, None, message=f"{str(exc).capitalize()}. Choose the label with keys 2-5.")
        if label is None:
            return answer(True, None, message="No sentence is needed.")
        if lookalike is not None and lookalike not in LOOKALIKES:
            return answer(False, label, message=f"Unknown look-alike {lookalike!r}.")
        if body.decision == "not_a_defect" and lookalike is None:
            return answer(False, label, message="Press q-u to say what it is.")
        words = clean_words(body.words)
        if words:
            return await from_words(label, words, data, headers, body.task_id)
        if label == "no_defect" and lookalike in LOOKALIKE_CAPTIONS:
            return answer(True, label, LOOKALIKE_CAPTIONS[lookalike], "lookalike")
        return await from_image(label, data, headers, body.task_id)

    @app.get("/road-api/health")
    async def health():
        return {"ok": True, "model": writer is not None, "send_image": settings.send_image}

    @app.post("/road-api/sentence")
    async def sentence(body: SentenceRequest, request: Request):
        if request.headers.get("x-road-captions") != "1":  # a cross-site form cannot set this header
            raise HTTPException(403, "Missing X-Road-Captions header.")
        headers = login_headers(request.headers)
        user = await current_user(headers)
        if user is None:
            raise HTTPException(401, "Log in to Label Studio first.")
        who = str(user.get("email") or user.get("id"))
        if not allowed(who):
            raise HTTPException(429, "Too many sentence requests. Wait one minute.")
        try:
            task = await ls.task(body.task_id, headers)
        except Denied as exc:
            raise HTTPException(exc.status, str(exc))
        data = task.get("data") or {}
        started = time.monotonic()
        result = await make_sentence(body, data, headers)
        write_log({
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "user": who,
            "task_id": body.task_id, "record_id": data.get("record_id"), "decision": body.decision,
            "lookalike": body.lookalike, "words": body.words, "prefetch": body.prefetch,
            "label": result["label"], "ok": result["ok"], "source": result["source"],
            "sentence": result["sentence"], "message": result["message"], "error": result["error"],
            "cached": result.get("cached", False), "ms": round(1000 * (time.monotonic() - started)),
            "model": settings.model_id or None,
        })
        return {k: result[k] for k in ("ok", "label", "sentence", "source", "message")}

    return app
