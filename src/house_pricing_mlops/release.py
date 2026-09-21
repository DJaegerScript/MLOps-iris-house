"""Validated, immutable metadata for one approved House model release."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType


class ReleaseValidationError(ValueError):
    """A release manifest does not satisfy the immutable release contract."""


_REQUIRED_FIELDS = {
    "model_name",
    "model_version",
    "registry_version",
    "dataset_version",
    "dataset_s3_bucket",
    "dataset_s3_key",
    "dataset_s3_version_id",
    "model_s3_bucket",
    "model_s3_key",
    "model_s3_version_id",
    "bundle_sha256",
    "mlflow_run_id",
    "selected_model",
    "metrics",
    "git_commit_sha",
    "status",
}
_OPTIONAL_FIELDS = {"approved_by", "approval_reason"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STATUSES = frozenset({"candidate", "champion"})


@dataclass(frozen=True, slots=True)
class HouseReleaseManifest:
    """The metadata needed to reproduce and deploy one model release.

    The manifest contains coordinates and evaluation metadata only. It never
    contains credentials, dataset rows, or model bytes.
    """

    model_name: str
    model_version: str
    registry_version: str
    dataset_version: str
    dataset_s3_bucket: str
    dataset_s3_key: str
    dataset_s3_version_id: str
    model_s3_bucket: str
    model_s3_key: str
    model_s3_version_id: str
    bundle_sha256: str
    mlflow_run_id: str
    selected_model: str
    metrics: Mapping[str, object]
    git_commit_sha: str
    status: str
    approved_by: str | None = None
    approval_reason: str | None = None

    def __post_init__(self) -> None:
        string_fields = (
            "model_name",
            "model_version",
            "registry_version",
            "dataset_version",
            "dataset_s3_bucket",
            "dataset_s3_key",
            "dataset_s3_version_id",
            "model_s3_bucket",
            "model_s3_key",
            "model_s3_version_id",
            "bundle_sha256",
            "mlflow_run_id",
            "selected_model",
            "git_commit_sha",
            "status",
        )
        for field in string_fields:
            _require_nonempty_string(getattr(self, field), field)

        if self.status not in _STATUSES:
            raise ReleaseValidationError(
                f"status must be one of {sorted(_STATUSES)}"
            )
        if not _SHA256_PATTERN.fullmatch(self.bundle_sha256):
            raise ReleaseValidationError("bundle_sha256 must be a SHA-256 checksum")

        mutable_fields = (
            "model_version",
            "dataset_version",
            "dataset_s3_bucket",
            "dataset_s3_key",
            "dataset_s3_version_id",
            "model_s3_bucket",
            "model_s3_key",
            "model_s3_version_id",
        )
        for field in mutable_fields:
            value = getattr(self, field)
            if "latest" in value.lower():
                raise ReleaseValidationError(
                    f"{field} cannot contain mutable latest reference"
                )

        if not isinstance(self.metrics, Mapping):
            raise ReleaseValidationError("metrics must be a JSON object")
        try:
            safe_metrics = json.loads(json.dumps(dict(self.metrics), sort_keys=True))
        except (TypeError, ValueError) as error:
            raise ReleaseValidationError("metrics must be JSON-safe") from error
        if not isinstance(safe_metrics, dict):  # pragma: no cover - defensive
            raise ReleaseValidationError("metrics must be a JSON object")
        object.__setattr__(self, "metrics", _freeze_json(safe_metrics))

        for field in ("approved_by", "approval_reason"):
            value = getattr(self, field)
            if value is not None:
                _require_nonempty_string(value, field)
        if self.status == "champion":
            if self.approved_by is None:
                raise ReleaseValidationError("approved_by is required for champion")
            if self.approval_reason is None:
                raise ReleaseValidationError(
                    "approval_reason is required for champion"
                )
        elif self.approved_by is not None or self.approval_reason is not None:
            raise ReleaseValidationError(
                "approval metadata is only valid for champion releases"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HouseReleaseManifest:
        """Validate one decoded JSON object and return an immutable manifest."""

        if not isinstance(payload, Mapping):
            raise ReleaseValidationError("release manifest must be a JSON object")
        missing = _REQUIRED_FIELDS - set(payload)
        if missing:
            raise ReleaseValidationError(
                f"release manifest is missing {sorted(missing)[0]}"
            )
        unexpected = set(payload) - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
        if unexpected:
            raise ReleaseValidationError(
                f"release manifest has unexpected field {sorted(unexpected)[0]}"
            )
        values = {field: payload[field] for field in _REQUIRED_FIELDS}
        values.update(
            {field: payload[field] for field in _OPTIONAL_FIELDS if field in payload}
        )
        try:
            return cls(**values)
        except TypeError as error:
            raise ReleaseValidationError(
                "release manifest fields are malformed"
            ) from error

    @classmethod
    def from_json(cls, value: str) -> HouseReleaseManifest:
        """Decode and validate one deterministic JSON manifest."""

        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise ReleaseValidationError(
                "release manifest JSON is malformed"
            ) from error
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe copy without credentials or raw artifact data."""

        payload: dict[str, object] = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "registry_version": self.registry_version,
            "dataset_version": self.dataset_version,
            "dataset_s3_bucket": self.dataset_s3_bucket,
            "dataset_s3_key": self.dataset_s3_key,
            "dataset_s3_version_id": self.dataset_s3_version_id,
            "model_s3_bucket": self.model_s3_bucket,
            "model_s3_key": self.model_s3_key,
            "model_s3_version_id": self.model_s3_version_id,
            "bundle_sha256": self.bundle_sha256,
            "mlflow_run_id": self.mlflow_run_id,
            "selected_model": self.selected_model,
            "metrics": _thaw_json(self.metrics),
            "git_commit_sha": self.git_commit_sha,
            "status": self.status,
        }
        if self.approved_by is not None:
            payload["approved_by"] = self.approved_by
        if self.approval_reason is not None:
            payload["approval_reason"] = self.approval_reason
        return payload

    def to_json(self) -> str:
        """Serialize the manifest deterministically for artifact handoff."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def with_approval(
        self, *, approved_by: str, approval_reason: str
    ) -> HouseReleaseManifest:
        """Return a champion copy with explicit approval metadata."""

        return replace(
            self,
            status="champion",
            approved_by=approved_by,
            approval_reason=approval_reason,
        )


@dataclass(frozen=True, slots=True)
class HouseReleaseDraft:
    """Pre-upload release metadata emitted by the training stage.

    Training knows the input dataset VersionId, but the model VersionId is
    assigned only by S3 after the bundle upload. This draft is not deployable;
    registration must finalize it as a :class:`HouseReleaseManifest`.
    """

    model_name: str
    model_version: str
    dataset_version: str
    dataset_s3_bucket: str
    dataset_s3_key: str
    dataset_s3_version_id: str
    model_s3_bucket: str
    model_s3_key: str
    model_s3_version_id: str | None
    bundle_sha256: str
    mlflow_run_id: str
    selected_model: str
    metrics: Mapping[str, object]
    git_commit_sha: str

    def __post_init__(self) -> None:
        for field in (
            "model_name",
            "model_version",
            "dataset_version",
            "dataset_s3_bucket",
            "dataset_s3_key",
            "dataset_s3_version_id",
            "model_s3_bucket",
            "model_s3_key",
            "bundle_sha256",
            "mlflow_run_id",
            "selected_model",
            "git_commit_sha",
        ):
            _require_nonempty_string(getattr(self, field), field)
        if not _SHA256_PATTERN.fullmatch(self.bundle_sha256):
            raise ReleaseValidationError("bundle_sha256 must be a SHA-256 checksum")
        for field in (
            "model_version",
            "dataset_version",
            "dataset_s3_bucket",
            "dataset_s3_key",
            "dataset_s3_version_id",
            "model_s3_bucket",
            "model_s3_key",
        ):
            if "latest" in getattr(self, field).lower():
                raise ReleaseValidationError(
                    f"{field} cannot contain mutable latest reference"
                )
        if self.model_s3_version_id is not None and "latest" in (
            self.model_s3_version_id.lower()
        ):
            raise ReleaseValidationError(
                "model_s3_version_id cannot contain mutable latest reference"
            )
        if not isinstance(self.metrics, Mapping):
            raise ReleaseValidationError("metrics must be a JSON object")
        try:
            safe_metrics = json.loads(json.dumps(dict(self.metrics), sort_keys=True))
        except (TypeError, ValueError) as error:
            raise ReleaseValidationError("metrics must be JSON-safe") from error
        object.__setattr__(self, "metrics", _freeze_json(safe_metrics))

    def to_dict(self) -> dict[str, object]:
        """Return the non-deployable draft in the release artifact shape."""

        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "registry_version": None,
            "dataset_version": self.dataset_version,
            "dataset_s3_bucket": self.dataset_s3_bucket,
            "dataset_s3_key": self.dataset_s3_key,
            "dataset_s3_version_id": self.dataset_s3_version_id,
            "model_s3_bucket": self.model_s3_bucket,
            "model_s3_key": self.model_s3_key,
            "model_s3_version_id": self.model_s3_version_id,
            "bundle_sha256": self.bundle_sha256,
            "mlflow_run_id": self.mlflow_run_id,
            "selected_model": self.selected_model,
            "metrics": _thaw_json(self.metrics),
            "git_commit_sha": self.git_commit_sha,
            "status": "training",
        }

    def to_json(self) -> str:
        """Serialize the draft deterministically for CodePipeline handoff."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def _require_nonempty_string(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseValidationError(f"{field} must be a non-empty string")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(nested) for key, nested in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "HouseReleaseDraft",
    "HouseReleaseManifest",
    "ReleaseValidationError",
]
