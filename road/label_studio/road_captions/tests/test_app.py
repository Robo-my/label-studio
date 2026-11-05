"""Service tests with a fake Label Studio (httpx MockTransport) and a fake model. No network."""
import asyncio
import json
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from road_captions.app import Settings, create_app

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 100
TASKS = {
    1: {"record_id": "r1", "detector_label": "pothole", "crop": "/data/local-files/?d=crops/r1.jpg"},
    2: {"record_id": "r2", "detector_label": "manhole", "crop": "/data/local-files/?d=crops/r2.jpg"},
    3: {"record_id": "r3", "detector_label": "raveling", "crop": "/tasks/3/presign/?fileuri=czM6Ly9iL3IzLmpwZw"},
    4: {"record_id": "r4", "detector_label": "speed_bump", "crop": "/data/local-files/?d=crops/r4.jpg"},
}
GOOD = {"Cookie": "sessionid=good", "X-Road-Captions": "1"}


class FakeLS:
    def __init__(self):
        self.seen = []

    def __call__(self, request):
        self.seen.append(request)
        logged_in = "sessionid=good" in request.headers.get("cookie", "")
        path = request.url.path
        if request.url.host == "s3.example":
            return httpx.Response(200, content=JPEG)
        if not logged_in:
            return httpx.Response(401, json={"detail": "no"})
        if path == "/api/current-user/whoami":
            return httpx.Response(200, json={"id": 7, "email": "intern07@example.com"})
        if path.startswith("/api/tasks/"):
            task_id = int(path.strip("/").split("/")[-1])
            if task_id == 99:
                return httpx.Response(403, json={"detail": "forbidden"})
            if task_id not in TASKS:
                return httpx.Response(404, json={"detail": "not found"})
            return httpx.Response(200, json={"id": task_id, "data": TASKS[task_id]})
        if path == "/data/local-files/":
            return httpx.Response(200, content=JPEG)
        if path == "/tasks/3/presign/":
            return httpx.Response(302, headers={"location": "https://s3.example/b/r3.jpg?X-Amz-Signature=abc"})
        return httpx.Response(404)


class FakeWriter:
    def __init__(self, reply="Pothole with broken edges.", delay=0.0, fail=False):
        self.reply, self.delay, self.fail = reply, delay, fail
        self.calls = []
        self.lock = threading.Lock()

    def write(self, prompt, image=None, image_format=None):
        with self.lock:
            self.calls.append({"prompt": prompt, "image": image, "format": image_format})
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("Bedrock is down")
        return self.reply(prompt) if callable(self.reply) else self.reply


def make(tmp_path, writer=None, **overrides):
    settings = Settings(ls_url="http://label-studio:8080", model_id="fake-model" if writer else "",
                        send_image=True, log_path=str(tmp_path / "requests.jsonl"), **overrides)
    fake_ls = FakeLS()
    app = create_app(settings, writer=writer, ls_transport=httpx.MockTransport(fake_ls))
    return TestClient(app), fake_ls


def ask(client, headers=GOOD, **body):
    return client.post("/road-api/sentence", json=body, headers=headers)


def test_needs_the_custom_header_and_a_login(tmp_path):
    client, _ = make(tmp_path)
    assert ask(client, headers={"Cookie": "sessionid=good"}, task_id=1, decision="Correct").status_code == 403
    assert ask(client, headers={"X-Road-Captions": "1"}, task_id=1, decision="Correct").status_code == 401
    bad = {"Cookie": "sessionid=bad", "X-Road-Captions": "1"}
    assert ask(client, headers=bad, task_id=1, decision="Correct").status_code == 401


def test_task_access_is_decided_by_label_studio(tmp_path):
    client, _ = make(tmp_path)
    assert ask(client, task_id=404, decision="Correct").status_code == 404
    assert ask(client, task_id=99, decision="Correct").status_code == 403


def test_model_sentence_from_the_crop_is_cached(tmp_path):
    writer = FakeWriter()
    client, _ = make(tmp_path, writer)
    first = ask(client, task_id=1, decision="Correct").json()
    assert first == {"ok": True, "label": "pothole", "sentence": "Pothole with broken edges.", "source": "bedrock",
                     "message": ""}
    assert writer.calls[0]["image"] == JPEG and writer.calls[0]["format"] == "jpeg"
    assert 'include the word "pothole"' in writer.calls[0]["prompt"]
    again = ask(client, task_id=1, decision="pothole").json()
    assert again["sentence"] == first["sentence"] and len(writer.calls) == 1


def test_two_requests_at_once_make_one_model_call(tmp_path):
    writer = FakeWriter(delay=0.3)
    settings = Settings(model_id="fake", send_image=True, log_path=str(tmp_path / "log.jsonl"))
    app = create_app(settings, writer=writer, ls_transport=httpx.MockTransport(FakeLS()))

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            body = {"task_id": 1, "decision": "Correct"}
            return await asyncio.gather(*[c.post("/road-api/sentence", json=body, headers=GOOD) for _ in range(3)])

    results = asyncio.run(run())
    assert [r.json()["sentence"] for r in results] == ["Pothole with broken edges."] * 3
    assert len(writer.calls) == 1


def test_bad_model_reply_falls_back_to_the_template(tmp_path):
    client, _ = make(tmp_path, FakeWriter(reply="Severe pothole, dangerous for bikes."))
    res = ask(client, task_id=1, decision="Correct").json()
    assert res["ok"] and res["source"] == "template" and res["sentence"] == "Pothole in the road surface."
    log = [json.loads(line) for line in open(tmp_path / "requests.jsonl")]
    assert "breaks the caption rules" in log[-1]["error"]


def test_model_failure_falls_back_and_is_not_cached(tmp_path):
    writer = FakeWriter(fail=True)
    client, _ = make(tmp_path, writer)
    assert ask(client, task_id=3, decision="Correct").json()["source"] == "template"
    writer.fail = False
    writer.reply = "Raveling: loose stones across the lane."
    assert ask(client, task_id=3, decision="Correct").json()["source"] == "bedrock"


def test_presigned_image_never_gets_the_login(tmp_path):
    writer = FakeWriter(reply="Raveling: loose stones across the lane.")
    client, fake_ls = make(tmp_path, writer)
    assert ask(client, task_id=3, decision="Correct").json()["source"] == "bedrock"
    s3_requests = [r for r in fake_ls.seen if r.url.host == "s3.example"]
    assert len(s3_requests) == 1 and "cookie" not in s3_requests[0].headers


def test_send_image_off_keeps_the_crop_in_region(tmp_path):
    writer = FakeWriter(reply="Pothole with water inside and broken edges.")
    settings = Settings(model_id="fake", send_image=False, rewrite_sends_image=True,
                        log_path=str(tmp_path / "log.jsonl"))
    app = create_app(settings, writer=writer, ls_transport=httpx.MockTransport(FakeLS()))
    with TestClient(app) as client:
        assert ask(client, task_id=1, decision="Correct").json()["source"] == "template"
        assert ask(client, task_id=1, decision="pothole", words="water inside, broken edges").json()["ok"]
    assert len(writer.calls) == 1 and writer.calls[0]["image"] is None


def test_rewrite_from_the_interns_words(tmp_path):
    writer = FakeWriter(reply="Pothole with water inside and broken edges.")
    client, _ = make(tmp_path, writer)
    res = ask(client, task_id=1, decision="Correct", words="water inside, broken edges").json()
    assert res == {"ok": True, "label": "pothole", "sentence": "Pothole with water inside and broken edges.",
                   "source": "bedrock", "message": ""}
    assert writer.calls[0]["image"] is None  # rewrites send words only by default
    assert 'Reviewer\'s words: "water inside, broken edges"' in writer.calls[0]["prompt"]


def test_rewrite_that_drops_a_word_uses_the_template(tmp_path):
    client, _ = make(tmp_path, FakeWriter(reply="Pothole with water inside."))
    res = ask(client, task_id=1, decision="Correct", words="water inside, broken edges").json()
    assert res["source"] == "template" and res["sentence"] == "Pothole: water inside, broken edges."


def test_words_that_name_another_defect_are_refused(tmp_path):
    writer = FakeWriter()
    client, _ = make(tmp_path, writer)
    res = ask(client, task_id=1, decision="crack", words="pothole edge").json()
    assert not res["ok"] and "pothole" in res["message"] and writer.calls == []


def test_a_valid_typed_sentence_is_used_as_typed(tmp_path):
    writer = FakeWriter()
    client, _ = make(tmp_path, writer)
    res = ask(client, task_id=1, decision="Correct", words="pothole with water inside").json()
    assert res["source"] == "intern" and res["sentence"] == "Pothole with water inside." and writer.calls == []


def test_lookalikes(tmp_path):
    writer = FakeWriter(reply="No defect: oil stain on the asphalt.")
    client, _ = make(tmp_path, writer)
    res = ask(client, task_id=2, decision="Correct").json()
    assert res["source"] == "lookalike" and res["sentence"] == "No defect: manhole cover on intact asphalt."
    res = ask(client, task_id=1, decision="not_a_defect", lookalike="shadow").json()
    assert res["sentence"] == "No defect: shadow on smooth asphalt."
    res = ask(client, task_id=1, decision="not_a_defect").json()
    assert not res["ok"] and res["message"] == "Press q-u to say what it is."
    res = ask(client, task_id=1, decision="not_a_defect", lookalike="other").json()
    assert res["source"] == "bedrock" and res["sentence"] == "No defect: oil stain on the asphalt."
    assert 'start with "No defect:"' in writer.calls[-1]["prompt"]


def test_other_lookalike_without_a_model_asks_for_words(tmp_path):
    client, _ = make(tmp_path)
    res = ask(client, task_id=1, decision="not_a_defect", lookalike="other").json()
    assert not res["ok"] and "2-6 words" in res["message"]
    res = ask(client, task_id=1, decision="not_a_defect", lookalike="other", words="oil stain").json()
    assert res["ok"] and res["sentence"] == "No defect: oil stain." and res["source"] == "template"


def test_keys_without_a_sentence_and_unknown_values(tmp_path):
    client, _ = make(tmp_path)
    assert ask(client, task_id=1, decision="bad_box").json() == {
        "ok": True, "label": None, "sentence": "", "source": "none", "message": "No sentence is needed."}
    res = ask(client, task_id=4, decision="Correct").json()
    assert not res["ok"] and "speed_bump" in res["message"]
    assert not ask(client, task_id=1, decision="maybe").json()["ok"]
    assert ask(client, task_id=1, decision="Correct", extra="x").status_code == 422


def test_rate_limit_per_intern(tmp_path):
    client, _ = make(tmp_path, rate_per_min=3)
    codes = [ask(client, task_id=1, decision="bad_box").status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_log_has_one_line_per_request(tmp_path):
    client, _ = make(tmp_path, FakeWriter())
    ask(client, task_id=1, decision="Correct", prefetch=True)
    ask(client, task_id=1, decision="Correct")
    rows = [json.loads(line) for line in open(tmp_path / "requests.jsonl")]
    assert [r["cached"] for r in rows] == [False, True]
    assert rows[0]["user"] == "intern07@example.com" and rows[0]["record_id"] == "r1" and rows[0]["prefetch"]


def test_health_and_static_module(tmp_path):
    client, _ = make(tmp_path)
    assert client.get("/road-api/health").json() == {"ok": True, "model": False, "send_image": True}
    js = client.get("/road-api/static/road-captions.js")
    assert js.status_code == 200 and "window.Htx" in js.text
