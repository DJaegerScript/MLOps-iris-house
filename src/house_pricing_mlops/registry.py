"""Local, approval-gated registry metadata for House Pricing artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from house_pricing_mlops.release import HouseReleaseManifest, ReleaseValidationError


class RegistryError(ValueError):
    """A safe model registration or promotion failure."""


class VersionedModelRegistry:
    """Persist explicit model versions and aliases in a local JSON registry.

    The file-backed adapter gives tests and local development the same
    approval semantics expected from a managed registry. Every record retains
    an immutable S3 object key and VersionId; there is intentionally no
    ``latest`` alias.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        default_model_name: str = "house-price-model",
    ) -> None:
        self.path = Path(path)
        self.default_model_name = default_model_name

    def register_candidate(
        self,
        *,
        model_name: str,
        model_version: str,
        run_id: str,
        dataset_version: str,
        dataset_sha256: str,
        bundle_sha256: str,
        s3_bucket: str,
        s3_key: str,
        s3_version_id: str,
        selected_model: str,
        metrics: Mapping[str, object],
        registered_at: str | None = None,
    ) -> dict[str, object]:
        """Register one selected run as a candidate without promotion."""

        _require_nonempty(
            model_name,
            model_version,
            run_id,
            dataset_version,
            dataset_sha256,
            bundle_sha256,
            s3_bucket,
            s3_key,
            s3_version_id,
            selected_model,
        )
        if model_name != self.default_model_name:
            raise RegistryError(f"model name must be {self.default_model_name}")
        if "latest" in s3_key.lower() or not s3_version_id.strip():
            raise RegistryError("registration requires an exact S3 VersionId")

        state = self._load()
        registry_version = str(len(state["versions"]) + 1)
        record: dict[str, object] = {
            "registry_version": registry_version,
            "model_name": model_name,
            "model_version": model_version,
            "status": "candidate",
            "run_id": run_id,
            "dataset_version": dataset_version,
            "dataset_sha256": dataset_sha256,
            "bundle_sha256": bundle_sha256,
            "s3_object": {
                "bucket": s3_bucket,
                "key": s3_key,
                "version_id": s3_version_id,
            },
            "selected_model": selected_model,
            "metrics": _json_safe(metrics),
            "registered_at": registered_at or datetime.now(UTC).isoformat(),
        }
        state["versions"].append(record)
        state["aliases"]["candidate"] = registry_version
        self._write(state)
        return record

    def register_candidate_release(
        self,
        *,
        model_name: str,
        model_version: str,
        run_id: str,
        dataset_version: str,
        dataset_s3_bucket: str,
        dataset_s3_key: str,
        dataset_s3_version_id: str,
        bundle_sha256: str,
        model_s3_bucket: str,
        model_s3_key: str,
        model_s3_version_id: str,
        selected_model: str,
        metrics: Mapping[str, object],
        git_commit_sha: str,
    ) -> HouseReleaseManifest:
        """Persist a complete candidate manifest after immutable S3 upload."""

        state = self._load()
        registry_version = str(len(state["versions"]) + 1)
        try:
            manifest = HouseReleaseManifest(
                model_name=model_name,
                model_version=model_version,
                registry_version=registry_version,
                dataset_version=dataset_version,
                dataset_s3_bucket=dataset_s3_bucket,
                dataset_s3_key=dataset_s3_key,
                dataset_s3_version_id=dataset_s3_version_id,
                model_s3_bucket=model_s3_bucket,
                model_s3_key=model_s3_key,
                model_s3_version_id=model_s3_version_id,
                bundle_sha256=bundle_sha256,
                mlflow_run_id=run_id,
                selected_model=selected_model,
                metrics=metrics,
                git_commit_sha=git_commit_sha,
                status="candidate",
            )
        except ReleaseValidationError as error:
            raise RegistryError(str(error)) from error

        record: dict[str, object] = {
            "registry_version": registry_version,
            "model_name": model_name,
            "model_version": model_version,
            "status": "candidate",
            "run_id": run_id,
            "dataset_version": dataset_version,
            "bundle_sha256": bundle_sha256,
            "selected_model": selected_model,
            "metrics": manifest.to_dict()["metrics"],
            "release_manifest": manifest.to_dict(),
            "registered_at": datetime.now(UTC).isoformat(),
        }
        state["versions"].append(record)
        state["aliases"]["candidate"] = registry_version
        self._write(state)
        return manifest

    def read_release_manifest(self, registry_version: str) -> HouseReleaseManifest:
        """Read and validate the complete release manifest for one version."""

        record = _find_version(self._load(), registry_version)
        payload = record.get("release_manifest")
        if not isinstance(payload, Mapping):
            raise RegistryError("registry version has no release manifest")
        try:
            return HouseReleaseManifest.from_dict(payload)
        except ReleaseValidationError as error:
            raise RegistryError(str(error)) from error

    def promote_champion(
        self,
        registry_version: str,
        *,
        approved_by: str,
        reason: str,
        approved_at: str | None = None,
    ) -> dict[str, object]:
        """Assign the champion alias only with an explicit decision record."""

        _require_approval(approved_by, reason)
        state = self._load()
        record = _find_version(state, registry_version)
        previous = state["aliases"].get("champion")
        record["status"] = "champion"
        _apply_release_approval(record, approved_by=approved_by, reason=reason)
        state["aliases"]["champion"] = registry_version
        state["promotion_decisions"].append(
            _decision(
                action="promote",
                promoted_version=registry_version,
                previous_champion=previous,
                approved_by=approved_by,
                reason=reason,
                approved_at=approved_at,
            )
        )
        self._write(state)
        return record

    def rollback_champion(
        self,
        registry_version: str,
        *,
        approved_by: str,
        reason: str,
        approved_at: str | None = None,
    ) -> dict[str, object]:
        """Point champion to a prior immutable version with an audit record."""

        _require_approval(approved_by, reason)
        state = self._load()
        record = _find_version(state, registry_version)
        previous = state["aliases"].get("champion")
        record["status"] = "champion"
        _apply_release_approval(record, approved_by=approved_by, reason=reason)
        state["aliases"]["champion"] = registry_version
        state["promotion_decisions"].append(
            _decision(
                action="rollback",
                promoted_version=registry_version,
                previous_champion=previous,
                approved_by=approved_by,
                reason=reason,
                approved_at=approved_at,
            )
        )
        self._write(state)
        return record

    def read(self) -> dict[str, Any]:
        """Read a JSON-safe snapshot of registry state."""

        return self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "model_name": self.default_model_name,
                "versions": [],
                "aliases": {},
                "promotion_decisions": [],
            }
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RegistryError("registry state is unreadable") from error
        if not isinstance(value, dict) or not isinstance(value.get("versions"), list):
            raise RegistryError("registry state is invalid")
        value.setdefault("model_name", self.default_model_name)
        value.setdefault("aliases", {})
        value.setdefault("promotion_decisions", [])
        return value

    def _write(self, state: Mapping[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _find_version(state: Mapping[str, Any], registry_version: str) -> dict[str, object]:
    for record in state["versions"]:
        if record.get("registry_version") == registry_version:
            return record
    raise RegistryError(f"unknown registry version: {registry_version}")


def _require_nonempty(*values: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise RegistryError("registration metadata is incomplete")


def _require_approval(approved_by: str, reason: str) -> None:
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise RegistryError("approved_by is required")
    if not isinstance(reason, str) or not reason.strip():
        raise RegistryError("promotion reason is required")


def _decision(
    *,
    action: str,
    promoted_version: str,
    previous_champion: object,
    approved_by: str,
    reason: str,
    approved_at: str | None,
) -> dict[str, object]:
    return {
        "action": action,
        "promoted_version": promoted_version,
        "previous_champion": previous_champion,
        "approved_by": approved_by,
        "reason": reason,
        "approved_at": approved_at or datetime.now(UTC).isoformat(),
    }


def _apply_release_approval(
    record: dict[str, object], *, approved_by: str, reason: str
) -> None:
    payload = record.get("release_manifest")
    if not isinstance(payload, Mapping):
        return
    try:
        manifest = HouseReleaseManifest.from_dict(payload).with_approval(
            approved_by=approved_by,
            approval_reason=reason,
        )
    except ReleaseValidationError as error:
        raise RegistryError(str(error)) from error
    record["release_manifest"] = manifest.to_dict()
    record["approved_by"] = approved_by
    record["approval_reason"] = reason


def _json_safe(value: Mapping[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(dict(value)))


__all__ = ["RegistryError", "VersionedModelRegistry"]
