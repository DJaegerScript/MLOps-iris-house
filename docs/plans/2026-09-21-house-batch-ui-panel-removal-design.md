# House Pricing Batch UI Panel Removal Design

Date: 2026-09-21

## Goal

Hide the visible Batch prediction panel from the House Pricing Streamlit page
while preserving the existing batch prediction implementation and backend
behavior.

## Scope

The change is limited to the House Pricing page render path in `app.py`:

- Remove the Batch prediction heading, CSV uploader, batch action button,
  result output, download control, metrics, drift diagnostics, and warning
  from the page.
- Keep `HouseBatchPredictionService`, batch validation, prediction, metrics,
  drift calculations, logging, and related tests unchanged.
- Preserve the single-property prediction flow and the Iris page unchanged.

## Alternatives considered

1. **Hide the panel with CSS or injected markup.** This keeps the render code
   active but is brittle and can leave inaccessible controls in the page.
2. **Add a runtime feature flag.** This supports future toggling but adds
   configuration and deployment complexity for a simple visibility change.
3. **Remove the batch render block while retaining the backend service.**
   Recommended: this cleanly hides the panel without changing batch behavior
   or deleting reusable functionality.

## Testing and acceptance criteria

- The House page boundary test verifies the batch uploader and Batch
  prediction heading are not rendered.
- Existing single-property House page behavior remains covered and passing.
- Batch service tests remain unchanged and passing.
- `git diff --check` is clean and the focused test suite passes.
- The final diff contains only the page-render change, its test coverage, and
  this design/implementation documentation.
