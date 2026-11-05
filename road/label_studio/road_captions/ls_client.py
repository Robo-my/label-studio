"""Calls from road-captions to Label Studio, made with the intern's own login.

road-captions never stores the login. It sends the intern's cookie to Label Studio to ask two things:
who is this (whoami), and may this person open this task (the task API). Label Studio decides.
"""
import asyncio
from urllib.parse import urljoin, urlparse

import httpx

FORWARD = ("cookie", "authorization")


class Denied(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def login_headers(request_headers):
    return {k: v for k, v in request_headers.items() if k.lower() in FORWARD}


class LabelStudio:
    def __init__(self, base_url, transport=None, timeout=10.0):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.AsyncClient(base_url=self.base_url, transport=transport, timeout=timeout,
                                      follow_redirects=False)
        # For presigned storage URLs on other hosts. It never gets the intern's login.
        self.outside = httpx.AsyncClient(transport=transport, timeout=timeout, follow_redirects=True)

    async def close(self):
        await self.http.aclose()
        await self.outside.aclose()

    async def whoami(self, headers):
        """Return the user dict, or None when the request is not logged in."""
        if not headers:
            return None
        r = await self.http.get("/api/current-user/whoami", headers=headers)
        if r.status_code != 200:
            return None
        user = r.json()
        return user if isinstance(user, dict) and user.get("id") is not None else None

    async def task(self, task_id, headers):
        r = await self.http.get(f"/api/tasks/{int(task_id)}/", headers=headers)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            raise Denied(404, "Task not found.")
        raise Denied(403, "Label Studio did not allow this task.")

    async def image(self, value, headers, max_bytes, s3_reader=None):
        """Get the crop bytes for a task data value. Returns (bytes, format)."""
        value = (value or "").strip()
        if value.startswith("s3://"):
            if s3_reader is None:
                raise ValueError("s3:// path, but ALLOWED_S3_BUCKETS is not set")
            data = await asyncio.to_thread(s3_reader.read, value, max_bytes)
            return data, image_format(data)
        url = self._label_studio_url(value)
        data = await self._get(url, headers, max_bytes)
        return data, image_format(data)

    def _label_studio_url(self, value):
        """Only follow URLs that Label Studio itself serves (presigned storage, local files, uploads)."""
        parsed = urlparse(value)
        if parsed.scheme in ("http", "https"):
            value = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        if not value.startswith(("/tasks/", "/projects/", "/data/")):
            raise ValueError(f"image URL is not served by Label Studio: {value[:80]!r}")
        return value

    async def _get(self, url, headers, max_bytes):
        r = await self.http.get(url, headers=headers)
        if r.is_redirect:
            target = urljoin(self.base_url + "/", r.headers["location"])
            if target.startswith(self.base_url):  # still Label Studio: keep the login
                r = await self.http.get(target, headers=headers)
            else:  # presigned storage URL: never send the login to another host
                r = await self.outside.get(target)
        if r.status_code != 200:
            raise ValueError(f"image request failed with HTTP {r.status_code}")
        if len(r.content) > max_bytes:
            raise ValueError("image is too large")
        return r.content


def image_format(data):
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:4] == b"GIF8":
        return "gif"
    raise ValueError("the crop is not a JPEG, PNG, WEBP or GIF image")
