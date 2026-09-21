"""Tests for trusted House Pricing bundle loading boundaries."""

import tarfile
from pathlib import Path

import pytest

from house_pricing_mlops.bundle import (
    BundleIntegrityError,
    load_model_bundle,
    load_model_bundle_bytes,
)


def test_malformed_archive_is_rejected() -> None:
    with pytest.raises(BundleIntegrityError, match="archive"):
        load_model_bundle_bytes(b"not a gzip archive")


def test_symlink_archive_member_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "symlink.tar.gz"
    with tarfile.open(path, mode="w:gz") as archive:
        member = tarfile.TarInfo("model.joblib")
        member.type = tarfile.SYMTYPE
        member.linkname = "/tmp/untrusted"
        archive.addfile(member)

    with pytest.raises(BundleIntegrityError, match="link"):
        load_model_bundle(path)
