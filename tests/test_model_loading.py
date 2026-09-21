"""Tests for private S3 retrieval and verified in-memory model loading."""

from __future__ import annotations

import hashlib
import io
import tarfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, StandardScaler

import iris_mlops.model as model_module
from iris_mlops.manifest import ModelManifest, load_manifest
from iris_mlops.model import (
    ArtifactIntegrityError,
    ArtifactLoadError,
    load_model_from_s3,
    load_verified_model,
)
from iris_mlops.storage import S3ArtifactStore

REPOSITORY_ROOT = Path(__file__).parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "artifacts" / "iris-classifier-v1.manifest.json"
ARTIFACT_NAMES = (
    "iris_model.pkl",
    "scaler.pkl",
    "label_encoder.pkl",
    "metadata.pkl",
)


def _untrained_objects() -> dict[str, object]:
    model = LinearDiscriminantAnalysis()
    model.classes_ = (0, 1, 2)
    model.n_features_in_ = 4

    scaler = StandardScaler()
    scaler.n_features_in_ = 4

    label_encoder = LabelEncoder()
    label_encoder.classes_ = (
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica",
    )
    return {
        "iris_model.pkl": model,
        "scaler.pkl": scaler,
        "label_encoder.pkl": label_encoder,
    }


def _bundle(
    members: dict[str, bytes],
    extra: list[tarfile.TarInfo] | None = None,
    *,
    member_mtime: int = 0,
) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = member_mtime
            archive.addfile(info, io.BytesIO(payload))
        for info in extra or []:
            archive.addfile(info)
    return stream.getvalue()


def _manifest_for(
    checksum_members: dict[str, bytes],
    bundle: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> ModelManifest:
    manifest = load_manifest(MANIFEST_PATH)
    checksums = {
        name: hashlib.sha256(checksum_members[name]).hexdigest()
        for name in ARTIFACT_NAMES
    }
    trusted = replace(
        model_module.V1_EXPECTATIONS,
        artifact_checksums=checksums,
        bundle_sha256=hashlib.sha256(bundle).hexdigest(),
    )
    monkeypatch.setattr(model_module, "V1_EXPECTATIONS", trusted)
    return replace(
        manifest,
        artifact_checksums=checksums,
        bundle_sha256=trusted.bundle_sha256,
    )


def _valid_members() -> dict[str, bytes]:
    return {name: f"synthetic-{name}".encode() for name in ARTIFACT_NAMES}


def test_s3_store_gets_only_the_explicit_versioned_object() -> None:
    body = io.BytesIO(b"private bundle")
    client = Mock()
    client.get_object.return_value = {"Body": body}
    store = S3ArtifactStore(client=client)

    assert store.get_versioned_object(
        "private-iris-bucket", "models/iris/v1.tar.gz", "version-123"
    ) == b"private bundle"
    client.get_object.assert_called_once_with(
        Bucket="private-iris-bucket",
        Key="models/iris/v1.tar.gz",
        VersionId="version-123",
    )
    client.assert_not_called()


def test_loader_checks_every_member_before_deserializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = _valid_members()
    tampered_members = dict(members)
    tampered_members["scaler.pkl"] = b"tampered"
    bundle = _bundle(tampered_members)
    manifest = _manifest_for(members, bundle, monkeypatch)
    deserializer = Mock()

    with pytest.raises(ArtifactIntegrityError, match="scaler.pkl"):
        load_verified_model(bundle, manifest, deserializer=deserializer)

    deserializer.assert_not_called()


def test_loader_checks_bundle_hash_before_member_deserialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = _valid_members()
    bundle = _bundle(members)
    manifest = _manifest_for(members, bundle, monkeypatch)
    deserializer = Mock()

    with pytest.raises(ArtifactIntegrityError, match="bundle checksum"):
        load_verified_model(
            _bundle(members, member_mtime=1), manifest, deserializer=deserializer
        )

    deserializer.assert_not_called()


def test_loader_deserializes_only_allowed_objects_from_in_memory_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = _valid_members()
    bundle = _bundle(members)
    manifest = _manifest_for(members, bundle, monkeypatch)
    objects = _untrained_objects()
    deserializer = Mock(side_effect=[objects[name] for name in objects])

    loaded = load_verified_model(bundle, manifest, deserializer=deserializer)

    assert loaded.model is objects["iris_model.pkl"]
    assert loaded.scaler is objects["scaler.pkl"]
    assert loaded.label_encoder is objects["label_encoder.pkl"]
    assert loaded.metadata is None
    assert deserializer.call_count == 3
    assert all(
        isinstance(call.args[0], io.BytesIO) for call in deserializer.call_args_list
    )
    assert not hasattr(objects["iris_model.pkl"], "coef_")
    assert not hasattr(objects["scaler.pkl"], "mean_")


@pytest.mark.parametrize(
    ("name", "member_factory", "message"),
    [
        (
            "/iris_model.pkl",
            lambda: None,
            "absolute path",
        ),
        (
            "nested/../iris_model.pkl",
            lambda: None,
            "path traversal",
        ),
        (
            "iris_model.pkl",
            lambda: tarfile.TarInfo("iris_model.pkl"),
            "symlink",
        ),
    ],
)
def test_loader_rejects_unsafe_tar_members(
    name, member_factory, message, monkeypatch: pytest.MonkeyPatch
) -> None:
    members = _valid_members()
    extra = []
    if message == "symlink":
        info = member_factory()
        info.type = tarfile.SYMTYPE
        info.linkname = "scaler.pkl"
        extra.append(info)
        members.pop(name)
    else:
        members.pop("iris_model.pkl")
        members[name] = b"unsafe"

    bundle = _bundle(members, extra)
    manifest = _manifest_for(_valid_members(), bundle, monkeypatch)

    with pytest.raises(ArtifactLoadError, match=message):
        load_verified_model(bundle, manifest)


def test_loader_rejects_unexpected_regular_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = _valid_members()
    unexpected = tarfile.TarInfo("unexpected.pkl")
    unexpected.size = 0
    bundle = _bundle(members, [unexpected])
    manifest = _manifest_for(members, bundle, monkeypatch)

    with pytest.raises(ArtifactLoadError, match="unexpected artifact member"):
        load_verified_model(bundle, manifest)


def test_loader_rejects_hardlinks_unexpected_members_and_missing_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = _valid_members()
    link = tarfile.TarInfo("unexpected.pkl")
    link.type = tarfile.LNKTYPE
    link.linkname = "iris_model.pkl"

    hardlink_bundle = _bundle(members, [link])
    hardlink_manifest = _manifest_for(members, hardlink_bundle, monkeypatch)
    with pytest.raises(ArtifactLoadError, match="hardlink"):
        load_verified_model(hardlink_bundle, hardlink_manifest)

    members.pop("metadata.pkl")
    missing_bundle = _bundle(members)
    missing_manifest = _manifest_for(_valid_members(), missing_bundle, monkeypatch)
    with pytest.raises(ArtifactLoadError, match="missing"):
        load_verified_model(missing_bundle, missing_manifest)


def test_loader_rejects_a_forged_manifest_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = _valid_members()
    bundle = _bundle(members)
    manifest = _manifest_for(members, bundle, monkeypatch)

    forged = replace(manifest, source_revision="forged-revision")

    with pytest.raises(ArtifactIntegrityError, match="trusted"):
        load_verified_model(bundle, forged)


def test_load_model_from_s3_wires_the_versioned_store_to_the_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = _valid_members()
    bundle = _bundle(members)
    manifest = _manifest_for(members, bundle, monkeypatch)
    objects = _untrained_objects()
    deserializer = Mock(side_effect=[objects[name] for name in objects])
    store = Mock()
    store.get_versioned_object.return_value = bundle

    loaded = load_model_from_s3(
        store,
        "private-iris-bucket",
        "models/iris/v1.tar.gz",
        "version-123",
        manifest,
        deserializer=deserializer,
    )

    store.get_versioned_object.assert_called_once_with(
        "private-iris-bucket", "models/iris/v1.tar.gz", "version-123"
    )
    assert loaded.model is objects["iris_model.pkl"]


def test_loader_rejects_a_manifest_that_is_not_the_validated_v1_type() -> None:
    members = _valid_members()

    with pytest.raises(ArtifactLoadError, match="validated v1 manifest"):
        load_verified_model(_bundle(members), {"artifact_checksums": {}})
