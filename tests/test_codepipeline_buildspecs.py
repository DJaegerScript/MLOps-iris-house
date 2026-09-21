"""Static contracts for the AWS CodeBuild House model stages."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
TRAIN_BUILDSPEC = ROOT / "buildspecs" / "house-model-train.yml"
PROMOTE_BUILDSPEC = ROOT / "buildspecs" / "house-model-promote.yml"
VALIDATOR = ROOT / "scripts" / "codepipeline" / "validate_house_release.py"


def test_training_buildspec_requires_immutable_inputs_and_publishes_candidate() -> None:
    assert TRAIN_BUILDSPEC.is_file(), "training buildspec is missing"
    buildspec = TRAIN_BUILDSPEC.read_text(encoding="utf-8")

    for variable in (
        "HOUSE_DATASET_S3_BUCKET",
        "HOUSE_DATASET_S3_KEY",
        "HOUSE_DATASET_S3_VERSION_ID",
        "DATASET_VERSION",
        "MODEL_VERSION",
    ):
        assert variable in buildspec
    assert "aws s3api get-object" in buildspec
    assert "--version-id" in buildspec
    assert "requirements-training.txt" in buildspec
    assert "scripts/train_house_price_model.py" in buildspec
    assert "--random-seed 42" in buildspec
    assert "aws s3api put-object" in buildspec
    assert "--query VersionId" in buildspec
    assert "register-candidate" in buildspec
    assert "release_manifest.json" in buildspec
    assert "artifacts:" in buildspec
    assert "secrets-manager" in buildspec or "parameter-store" in buildspec
    assert "update-express-gateway-service" not in buildspec
    assert "create-express-gateway-service" not in buildspec


def test_promotion_buildspec_validates_approval_and_writes_immutable_release() -> None:
    assert PROMOTE_BUILDSPEC.is_file(), "promotion buildspec is missing"
    buildspec = PROMOTE_BUILDSPEC.read_text(encoding="utf-8")

    assert "release_manifest.json" in buildspec
    assert "APPROVED_BY" in buildspec
    assert "APPROVAL_REASON" in buildspec
    assert "promote" in buildspec
    assert "promotion_decisions" in buildspec
    assert "mlflow/registry/house-price-model" in buildspec
    assert "aws s3api put-object" in buildspec
    assert "approved-release.json" in buildspec
    assert "validate_house_release.py" in buildspec
    assert "champion" in buildspec
    assert "update-express-gateway-service" not in buildspec


def test_release_validator_requires_finalized_champion_coordinates() -> None:
    assert VALIDATOR.is_file(), "release validator is missing"
    validator = VALIDATOR.read_text(encoding="utf-8")

    assert "HouseReleaseManifest" in validator
    assert "ReleaseValidationError" in validator
    assert "require_champion" in validator
    assert "VersionId" in validator
    assert "credentials" not in validator.lower()
