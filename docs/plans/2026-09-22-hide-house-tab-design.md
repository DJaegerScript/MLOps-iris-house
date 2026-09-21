# Hide House Pricing Tab in Iris Streamlit

**Status:** Approved

## Goal

Keep the Iris Streamlit surface focused on Iris by removing the House Pricing option from the Iris product navigation, while preserving the standalone House runtime used when `APP_VARIANT=house`.

## Approaches considered

1. **Restrict the Iris product options to Iris only (recommended).** This is a one-line behavior change at the existing navigation seam and leaves House code and its dedicated runtime untouched.
2. Hide the option with custom CSS. This would leave House selectable in the application state and depend on Streamlit markup, so it is brittle.
3. Remove the House page and its runtime. This exceeds the requested presentation-only scope and would break the separate House deployment.

## Design

Update `_selected_product()` in `app.py` so the `APP_VARIANT=iris` navigation offers only `IRIS_PRODUCT`. The existing `APP_VARIANT=house` branch remains unchanged and continues to render `_render_house_page()` without product navigation.

Add a focused regression assertion to the Streamlit boundary tests that verifies the Iris navigation options contain only the Iris product. Keep the existing House-runtime and Iris-serving tests intact.

## Verification

Run the focused application tests, the complete test suite, `git diff --check`, and inspect the final staged diff. Push the resulting commits directly to `origin/main` and verify the remote `main` ref contains the new commit.
