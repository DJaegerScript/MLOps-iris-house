"""Tests for the House Pricing package and test-data boundary."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
FIXTURE_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "house_prices" / "train.csv"


def test_house_pricing_package_imports() -> None:
    import house_pricing_mlops

    assert house_pricing_mlops.__doc__


def test_house_fixture_is_available_and_is_not_an_production_dataset() -> None:
    assert FIXTURE_PATH.is_file()
    assert "tests/fixtures" in FIXTURE_PATH.as_posix()


def test_house_fixture_has_expected_header() -> None:
    header = FIXTURE_PATH.read_text(encoding="utf-8").splitlines()[0].split(",")

    assert "Id" in header
    assert "SalePrice" in header
    assert "OverallQual" in header
    assert "Neighborhood" in header
