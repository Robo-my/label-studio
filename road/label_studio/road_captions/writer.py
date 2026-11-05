"""Bedrock writer: one Converse call that returns the model's reply text.

The Converse API works with any Bedrock model that reads text and images, so the model is a setting.
Credentials come from the EC2 instance role through the normal boto3 chain. Never put keys in .env.
"""
import boto3
from botocore.config import Config


class WriterError(Exception):
    """The model answered, but not with a usable reply."""


class BedrockWriter:
    def __init__(self, model_id, region, timeout_s=10.0, max_tokens=2048, extra_fields=None):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.extra_fields = extra_fields or None
        self.client = boto3.client(
            "bedrock-runtime", region_name=region,
            config=Config(connect_timeout=3, read_timeout=timeout_s, retries={"max_attempts": 2, "mode": "standard"}),
        )

    def write(self, prompt, image=None, image_format=None):
        """Send the prompt (and the crop, if given). Return the reply text. Raises on any failure."""
        content = []
        if image is not None:
            content.append({"image": {"format": image_format, "source": {"bytes": image}}})
        content.append({"text": prompt})
        # No temperature or top_p: newer models reject sampling settings.
        # maxTokens leaves room for models that think before they answer.
        kwargs = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "inferenceConfig": {"maxTokens": self.max_tokens},
        }
        if self.extra_fields:
            kwargs["additionalModelRequestFields"] = self.extra_fields
        resp = self.client.converse(**kwargs)
        stop = resp.get("stopReason")
        if stop not in ("end_turn", "stop_sequence"):
            raise WriterError(f"model stopped with {stop!r}")
        blocks = resp.get("output", {}).get("message", {}).get("content", [])
        return "\n".join(b["text"] for b in blocks if "text" in b)


class S3Reader:
    """Reads crops straight from S3 when the task data holds s3:// paths that Label Studio did not resolve."""

    def __init__(self, region, allowed_buckets):
        self.allowed = set(allowed_buckets)
        self.client = boto3.client("s3", region_name=region)

    def read(self, uri, max_bytes):
        bucket, _, key = uri[len("s3://"):].partition("/")
        if bucket not in self.allowed:
            raise PermissionError(f"bucket {bucket!r} is not in ALLOWED_S3_BUCKETS")
        body = self.client.get_object(Bucket=bucket, Key=key)["Body"]
        data = body.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("image is too large")
        return data
