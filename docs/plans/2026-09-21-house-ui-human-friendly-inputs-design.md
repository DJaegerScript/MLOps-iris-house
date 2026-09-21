# House Pricing Human-Friendly Inputs Design

## Goal

Make every House Pricing input label and visible categorical option readable for
users while preserving the original model feature keys and categorical values
sent to prediction.

## Scope

The change is limited to the House Pricing Streamlit page in `app.py` and its
page-boundary tests in `tests/test_house_app.py`. The model schema, validation
messages, persisted artifacts, and prediction service contract remain unchanged.

## Design

`app.py` will define explicit presentation metadata for all twelve model inputs.
Numeric fields will use labels such as `Above-Ground Living Area (sq ft)` and
`Total Rooms Above Grade`; categorical fields will use labels such as
`Kitchen Quality` and `Central Air Conditioning`.

Categorical options will remain raw values in the selectbox's returned value,
but Streamlit will render a display label through its option formatter. Known
House Prices codes will have domain-friendly names: `CentralAir` values `Y` and
`N` become `Yes` and `No`, kitchen quality codes become quality descriptions,
and neighborhood abbreviations become readable neighborhood names. A safe
fallback formatter will handle categories not present in the explicit map
without changing their raw values.

The page-boundary test will capture visible labels and formatter behavior, then
assert that the prediction service still receives raw values such as
`CentralAir="Y"`. Existing input-key and prediction assertions remain in place.

## Verification

Run the focused House UI test module and the full test suite. Inspect the diff
to confirm that only the approved presentation boundary and its tests changed,
apart from this design and implementation plan documentation.
