"""Private S3 access for version-pinned Iris model bundles."""

from __future__ import annotations

from typing import Any


class S3ArtifactStore:
    """Read one explicitly versioned object from S3.

    The client is created through boto3's normal ambient credential chain when
    one is not supplied. No credentials, public URLs, or bucket listings are
    part of this storage boundary.
    """

    def __init__(self, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            return

        try:
            import boto3
        except ModuleNotFoundError as error:  # pragma: no cover - packaging guard
            raise RuntimeError("boto3 is required to access S3 artifacts") from error

        self._client = boto3.client("s3")

    def get_versioned_object(self, bucket: str, key: str, version_id: str) -> bytes:
        """Return the bytes for exactly ``Bucket``, ``Key``, and ``VersionId``."""

        for field, value in (
            ("bucket", bucket),
            ("key", key),
            ("version_id", version_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")

        response = self._client.get_object(
            Bucket=bucket,
            Key=key,
            VersionId=version_id,
        )
        body = response["Body"]
        try:
            payload = body.read()
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()

        if not isinstance(payload, bytes):
            raise TypeError("S3 artifact body must return bytes")
        return payload


__all__ = ["S3ArtifactStore"]
