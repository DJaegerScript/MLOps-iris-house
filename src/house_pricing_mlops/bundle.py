"""Creation and checksum-verified loading of House Pricing model bundles."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import platform
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import joblib

from house_pricing_mlops.drift import build_reference_profile
from house_pricing_mlops.provenance import DatasetProvenance
from house_pricing_mlops.schema import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MODEL_SCHEMA_VERSION,
    NUMERIC_FEATURES,
    TARGET_NAME,
)
from house_pricing_mlops.training import TrainingResult

BUNDLE_MEMBERS = (
    "model.joblib",
    "manifest.json",
    "metrics.json",
    "reference_profile.json",
    "feature_schema.json",
    "environment.json",
)
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = len(BUNDLE_MEMBERS)
_CHECKSUM_PLACEHOLDER = "0" * 64


class BundleValidationError(ValueError):
    """Base error for invalid House Pricing bundle data."""


class BundleIntegrityError(BundleValidationError):
    """Bundle structure or checksum verification failed."""


class BundleLoadError(BundleValidationError):
    """A verified bundle could not be deserialized or used."""


@dataclass(frozen=True, slots=True)
class BundleArtifact:
    """Written bundle metadata returned by the creator."""

    path: Path
    bundle_sha256: str
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedHousePriceModel:
    """Trusted fitted pipeline and metadata loaded from one bundle."""

    pipeline: Any
    manifest: Mapping[str, Any]
    metrics: Mapping[str, Any]
    reference_profile: Mapping[str, Any]
    feature_schema: Mapping[str, Any]
    environment: Mapping[str, Any]

    def predict(self, features: Any) -> Any:
        """Delegate prediction to the complete fitted pipeline."""

        try:
            prediction = self.pipeline.predict(features)
        except Exception as error:
            raise BundleLoadError("verified model prediction failed") from error
        return prediction


def create_model_bundle(
    result: TrainingResult,
    provenance: DatasetProvenance,
    *,
    output_path: str | Path,
    model_version: str,
    model_name: str = "house-price-model",
) -> BundleArtifact:
    """Serialize the complete fitted pipeline and deterministic metadata bundle."""

    model_stream = io.BytesIO()
    joblib.dump(result.selected_pipeline, model_stream, compress=3)
    feature_schema = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_names": list(MODEL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_name": TARGET_NAME,
    }
    metrics = {
        "selected_model": result.selected_model_name,
        "validation_metrics": result.selected_candidate.validation_metrics,
        "test_metrics": result.test_metrics,
        "candidate_validation_metrics": {
            candidate.name: candidate.validation_metrics
            for candidate in result.candidates
        },
    }
    environment = _environment()
    members: dict[str, bytes] = {
        "model.joblib": model_stream.getvalue(),
        "metrics.json": _json_bytes(metrics),
        "reference_profile.json": _json_bytes(
            build_reference_profile(result.training_features)
        ),
        "feature_schema.json": _json_bytes(feature_schema),
        "environment.json": _json_bytes(environment),
    }
    artifact_checksums = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in members.items()
    }
    manifest: dict[str, Any] = {
        "model_name": model_name,
        "model_version": model_version,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "dataset_version": provenance.dataset_version,
        "dataset_sha256": provenance.dataset_sha256,
        "git_commit_sha": provenance.git_commit_sha,
        "feature_names": list(MODEL_FEATURES),
        "feature_types": {
            "numeric": list(NUMERIC_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
        },
        "target_name": TARGET_NAME,
        "model_type": result.selected_model_name,
        "framework_versions": environment,
        "training_parameters": {
            "configuration": result.config.to_dict(),
            "selected_model": result.selected_candidate.parameters,
        },
        "evaluation_metrics": metrics,
        "artifact_checksums": artifact_checksums,
        "bundle_sha256": _CHECKSUM_PLACEHOLDER,
        "training_timestamp": datetime.now(UTC).isoformat(),
        "source_url": provenance.source_url,
    }
    canonical_members = dict(members)
    canonical_members["manifest.json"] = _json_bytes(manifest)
    bundle_sha256 = _canonical_checksum(canonical_members)
    manifest["bundle_sha256"] = bundle_sha256
    all_members = dict(members)
    all_members["manifest.json"] = _json_bytes(manifest)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_archive(destination, all_members)
    return BundleArtifact(destination, bundle_sha256, manifest)


def load_model_bundle(path: str | Path) -> LoadedHousePriceModel:
    """Load and verify one local model bundle."""

    bundle_path = Path(path)
    try:
        bundle_bytes = bundle_path.read_bytes()
    except OSError as error:
        raise BundleLoadError("could not read model bundle") from error
    return load_model_bundle_bytes(bundle_bytes)


def load_model_bundle_bytes(bundle_bytes: bytes) -> LoadedHousePriceModel:
    """Verify and deserialize one complete bundle from memory."""

    if not isinstance(bundle_bytes, bytes):
        raise BundleLoadError("model bundle must be bytes")
    if len(bundle_bytes) > MAX_BUNDLE_BYTES:
        raise BundleIntegrityError("model bundle exceeds maximum size")
    members = _read_archive(bundle_bytes)
    manifest = _read_json(members["manifest.json"], "manifest.json")
    _validate_manifest(manifest)
    checksums = manifest["artifact_checksums"]
    for name in BUNDLE_MEMBERS:
        if name == "manifest.json":
            continue
        actual = hashlib.sha256(members[name]).hexdigest()
        if not hmac.compare_digest(actual, checksums[name]):
            raise BundleIntegrityError(f"checksum mismatch for {name}")
    canonical_members = dict(members)
    normalized_manifest = dict(manifest)
    normalized_manifest["bundle_sha256"] = _CHECKSUM_PLACEHOLDER
    canonical_members["manifest.json"] = _json_bytes(normalized_manifest)
    actual_bundle_sha256 = _canonical_checksum(canonical_members)
    if not hmac.compare_digest(actual_bundle_sha256, manifest["bundle_sha256"]):
        raise BundleIntegrityError("bundle checksum mismatch")
    try:
        pipeline = joblib.load(io.BytesIO(members["model.joblib"]))
    except Exception as error:
        raise BundleLoadError("could not deserialize verified model") from error
    if not callable(getattr(pipeline, "predict", None)):
        raise BundleLoadError("verified model does not provide predict")
    return LoadedHousePriceModel(
        pipeline=pipeline,
        manifest=manifest,
        metrics=_read_json(members["metrics.json"], "metrics.json"),
        reference_profile=_read_json(
            members["reference_profile.json"], "reference_profile.json"
        ),
        feature_schema=_read_json(
            members["feature_schema.json"], "feature_schema.json"
        ),
        environment=_read_json(members["environment.json"], "environment.json"),
    )


def _read_archive(bundle_bytes: bytes) -> dict[str, bytes]:
    try:
        archive = tarfile.open(fileobj=io.BytesIO(bundle_bytes), mode="r:gz")
    except Exception as error:
        raise BundleIntegrityError("could not open model bundle archive") from error
    found: dict[str, bytes] = {}
    try:
        with archive:
            for count, member in enumerate(archive, start=1):
                if count > MAX_ARCHIVE_MEMBERS:
                    raise BundleIntegrityError("too many archive members")
                _validate_member(member)
                if member.name in found:
                    raise BundleIntegrityError(
                        f"duplicate archive member: {member.name}"
                    )
                if member.name not in BUNDLE_MEMBERS:
                    raise BundleIntegrityError(
                        f"unexpected archive member: {member.name}"
                    )
                if member.size > MAX_MEMBER_BYTES:
                    raise BundleIntegrityError(
                        f"archive member exceeds size: {member.name}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise BundleIntegrityError(
                        f"could not read archive member: {member.name}"
                    )
                payload = extracted.read(MAX_MEMBER_BYTES + 1)
                if len(payload) > MAX_MEMBER_BYTES:
                    raise BundleIntegrityError(
                        f"archive member exceeds size: {member.name}"
                    )
                found[member.name] = payload
    except BundleIntegrityError:
        raise
    except Exception as error:
        raise BundleIntegrityError("could not read model bundle archive") from error
    missing = [name for name in BUNDLE_MEMBERS if name not in found]
    if missing:
        raise BundleIntegrityError(f"missing archive members: {', '.join(missing)}")
    return found


def _validate_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if (
        not member.name
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != member.name
    ):
        raise BundleIntegrityError(f"invalid archive path: {member.name}")
    if member.issym() or member.islnk():
        raise BundleIntegrityError(f"archive link is not allowed: {member.name}")
    if not member.isfile():
        raise BundleIntegrityError(f"archive member is not a file: {member.name}")


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "model_name",
        "model_version",
        "model_schema_version",
        "dataset_version",
        "dataset_sha256",
        "git_commit_sha",
        "feature_names",
        "feature_types",
        "target_name",
        "model_type",
        "framework_versions",
        "training_parameters",
        "evaluation_metrics",
        "artifact_checksums",
        "bundle_sha256",
        "training_timestamp",
        "source_url",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise BundleIntegrityError(
            f"manifest missing required fields: {', '.join(missing)}"
        )
    if manifest["model_schema_version"] != MODEL_SCHEMA_VERSION:
        raise BundleIntegrityError("manifest schema version is not trusted")
    if tuple(manifest["feature_names"]) != MODEL_FEATURES:
        raise BundleIntegrityError("manifest feature names are not trusted")
    if manifest["target_name"] != TARGET_NAME:
        raise BundleIntegrityError("manifest target is not trusted")
    checksums = manifest["artifact_checksums"]
    expected = {name for name in BUNDLE_MEMBERS if name != "manifest.json"}
    if set(checksums) != expected:
        raise BundleIntegrityError("manifest artifact checksum schema is invalid")


def _canonical_checksum(members: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(members):
        payload = members[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _write_archive(path: Path, members: Mapping[str, bytes]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for name in sorted(members):
            payload = members[name]
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _read_json(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleIntegrityError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise BundleIntegrityError(f"{name} must contain an object")
    return value


def _environment() -> dict[str, str]:
    import importlib.metadata

    packages = {}
    for package in ("joblib", "numpy", "pandas", "scikit-learn"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {"python": platform.python_version(), **packages}


__all__ = [
    "BUNDLE_MEMBERS",
    "BundleArtifact",
    "BundleIntegrityError",
    "BundleLoadError",
    "BundleValidationError",
    "LoadedHousePriceModel",
    "create_model_bundle",
    "load_model_bundle",
    "load_model_bundle_bytes",
]
