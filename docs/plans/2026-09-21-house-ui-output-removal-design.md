# House Pricing UI Output Removal Design

Date: 2026-09-21

## Goal

Remove the educational warning and the displayed test-evaluation metrics from
the House Pricing Streamlit page while preserving all model computation,
logging, batch evaluation, and documentation behavior.

## Scope

The change is limited to the House Pricing page render path in `app.py`:

- Remove the `st.info` warning about historical Ames data.
- Remove the `_render_house_metrics` call and its now-unused helper.
- Leave `test_metrics` generation, MLflow metric logging, batch metrics, and
  existing model artifacts unchanged.

The batch prediction section is outside this request and continues to display
its own optional evaluation and drift diagnostics.

## Alternatives considered

1. **Remove only the two render calls and leave dead helper code.** This is the
   smallest textual diff, but leaves an unused UI helper behind.
2. **Remove the warning and the helper plus its page call.** Recommended: it
   removes both requested outputs and keeps the render module clean without
   changing the underlying model contract.
3. **Add a configuration flag to hide the outputs.** This adds runtime
   complexity and an extra behavior not requested by the user.

## Testing and acceptance criteria

- A focused app test asserts the House Pricing page no longer emits the
  warning or the four test metric labels.
- The focused test is observed failing before the implementation change and
  passing afterward.
- The existing House app test suite and repository test suite are run as
  appropriate.
- The final diff contains only the approved UI change, its test coverage, and
  this design/implementation documentation.
