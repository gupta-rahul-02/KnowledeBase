"""S3-compatible blob store wrapper (targets Cloudflare R2). Keys: kbs/{kb_id}/{name}."""

import os
import threading
from typing import Optional

import boto3
from botocore.config import Config

_client = None
_client_lock = threading.Lock()


def _get_bucket() -> str:
    bucket = os.getenv("R2_BUCKET")
    if not bucket:
        raise RuntimeError("R2_BUCKET is not set")
    return bucket


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                endpoint = os.getenv("R2_ENDPOINT_URL")
                access_key = os.getenv("R2_ACCESS_KEY_ID")
                secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
                if not endpoint or not access_key or not secret_key:
                    raise RuntimeError(
                        "R2_ENDPOINT_URL / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY must be set"
                    )
                _client = boto3.client(
                    "s3",
                    endpoint_url=endpoint,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=os.getenv("R2_REGION", "auto"),
                    config=Config(signature_version="s3v4"),
                )
                print(f"[INFO] R2/S3 blob store client initialized ({endpoint})")
    return _client


def _key(kb_id: str, name: str) -> str:
    return f"kbs/{kb_id}/{name}"


def put(kb_id: str, name: str, data: bytes) -> None:
    _get_client().put_object(Bucket=_get_bucket(), Key=_key(kb_id, name), Body=data)


def get(kb_id: str, name: str) -> bytes:
    resp = _get_client().get_object(Bucket=_get_bucket(), Key=_key(kb_id, name))
    return resp["Body"].read()


def delete(kb_id: str, name: str) -> None:
    _get_client().delete_object(Bucket=_get_bucket(), Key=_key(kb_id, name))


def delete_kb(kb_id: str) -> int:
    """Delete all objects under kbs/{kb_id}/. Returns count deleted."""
    client = _get_client()
    bucket = _get_bucket()
    prefix = f"kbs/{kb_id}/"
    paginator = client.get_paginator("list_objects_v2")
    deleted = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents") or []
        if not contents:
            continue
        objects = [{"Key": obj["Key"]} for obj in contents]
        for i in range(0, len(objects), 1000):
            batch = objects[i : i + 1000]
            client.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
            deleted += len(batch)
    return deleted
