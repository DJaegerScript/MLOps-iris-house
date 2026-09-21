"""Tests for private S3 retrieval and verified in-memory model loading."""

from __future__ import annotations

import hashlib
import io
import tarfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from iris_mlops.manifest import ModelManifest, load_manifest
from iris_mlops.model import (
    ArtifactIntegrityError,
    ArtifactLoadError,
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


class LinearDiscriminantAnalysis:
    """Small test double named like the allowed serialized estimator."""

    classes_ = (0, 1, 2)
    n_features_in_ = 4

    def predict(self, features: object) -> list[int]:
        raise AssertionError("prediction is not part of this loading test")

    def predict_proba(self, features: object) -> list[list[float]]:
        raise AssertionError("prediction is not part of this loading test")


class StandardScaler:
    n_features_in_ = 4

    def transform(self, features: object) -> object:
        return features


class LabelEncoder:
    classes_ = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")

    def inverse_transform(self, values: list[int]) -> list[str]:
        return [self.classes_[value] for value in values]


class Metadata:
    pass


def _bundle(
    members: dict[str, bytes], extra: list[tarfile.TarInfo] | None = None
) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        for info in extra or []:
            archive.addfile(info)
    return stream.getvalue()


def _manifest_for(members: dict[str, bytes]) -> ModelManifest:
    manifest = load_manifest(MANIFEST_PATH)
    return replace(
        manifest,
        artifact_checksums={
            name: hashlib.sha256(members[name]).hexdigest() for name in ARTIFACT_NAMES
        },
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


def test_loader_checks_every_member_before_deserializing() -> None:
    members = _valid_members()
    manifest = _manifest_for(members)
    members["scaler.pkl"] = b"tampered"
    deserializer = Mock()

    with pytest.raises(ArtifactIntegrityError, match="scaler.pkl"):
        load_verified_model(_bundle(members), manifest, deserializer=deserializer)

    deserializer.assert_not_called()


def test_loader_deserializes_only_allowed_objects_from_in_memory_streams() -> None:
    members = _valid_members()
    manifest = _manifest_for(members)
    objects = {
        "iris_model.pkl": LinearDiscriminantAnalysis(),
        "scaler.pkl": StandardScaler(),
        "label_encoder.pkl": LabelEncoder(),
    }
    deserializer = Mock(side_effect=[objects[name] for name in objects])

    loaded = load_verified_model(_bundle(members), manifest, deserializer=deserializer)

    assert loaded.model is objects["iris_model.pkl"]
    assert loaded.scaler is objects["scaler.pkl"]
    assert loaded.label_encoder is objects["label_encoder.pkl"]
    assert loaded.metadata is None
    assert deserializer.call_count == 3
    assert all(
        isinstance(call.args[0], io.BytesIO) for call in deserializer.call_args_list
    )
    assert all(not hasattr(obj, "fit") for obj in objects.values())


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
def test_loader_rejects_unsafe_tar_members(name, member_factory, message) -> None:
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

    manifest = _manifest_for(_valid_members())

    with pytest.raises(ArtifactLoadError, match=message):
        load_verified_model(_bundle(members, extra), manifest)


def test_loader_rejects_hardlinks_unexpected_members_and_missing_members() -> None:
    members = _valid_members()
    link = tarfile.TarInfo("unexpected.pkl")
    link.type = tarfile.LNKTYPE
    link.linkname = "iris_model.pkl"

    with pytest.raises(ArtifactLoadError, match="hardlink"):
        load_verified_model(_bundle(members, [link]), _manifest_for(members))

    members.pop("metadata.pkl")
    with pytest.raises(ArtifactLoadError, match="missing"):
        load_verified_model(_bundle(members), _manifest_for(_valid_members()))


def test_loader_rejects_a_manifest_that_is_not_the_validated_v1_type() -> None:
    members = _valid_members()

    with pytest.raises(ArtifactLoadError, match="validated v1 manifest"):
        load_verified_model(_bundle(members), {"artifact_checksums": {}})
