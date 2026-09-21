# House Pricing Human-Friendly Inputs Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Render readable House Pricing field labels and categorical options while preserving the raw feature keys and category values consumed by the prediction service.

**Architecture:** Keep presentation metadata at the Streamlit page boundary in `app.py`. Numeric inputs use an explicit feature-label map; categorical selectboxes use Streamlit's `format_func` to display mapped text while returning the original option value. Tests observe both the visible presentation metadata and the raw prediction payload.

**Tech Stack:** Python 3.11+, Streamlit, pytest.

---

### Task 1: Add failing coverage for readable labels and raw option values

**Files:**
- Modify: `tests/test_house_app.py:20-70` to record input labels, options, and selectbox formatters.
- Modify: `tests/test_house_app.py:105-180` to assert the complete visible-label map, representative formatted options, and raw values passed to the prediction service.

**Step 1: Write the failing test**

Update the fake Streamlit `number_input` and `selectbox` methods to retain the
visible label and selectbox `format_func`. Set representative raw categorical
values (`CollgCr`, `TA`, and `Y`) in the fake input state. Add assertions that:

- all numeric and categorical field labels use readable text;
- `CollgCr`, `TA`, and `Y` render as readable option text;
- the service receives `Neighborhood="CollgCr"`, `KitchenQual="TA"`, and
  `CentralAir="Y"`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_house_app.py -q`

Expected: FAIL because the current page uses raw feature keys and does not pass
a categorical option formatter.

### Task 2: Implement the presentation mapping

**Files:**
- Modify: `app.py:35-50` to add explicit House Pricing feature and category presentation maps.
- Modify: `app.py:225-255` to use readable numeric/categorical labels and raw-preserving option formatters.
- Modify: `app.py` near `_house_category_options` to add the category display helper.

**Step 1: Add explicit feature labels**

Define labels for all twelve model inputs, including `Overall Quality`,
`Above-Ground Living Area (sq ft)`, `Garage Capacity (cars)`, `Total Basement
Area (sq ft)`, `First-Floor Area (sq ft)`, `Year Built`, `Full Bathrooms`,
`Total Rooms Above Grade`, `Garage Area (sq ft)`, `Neighborhood`, `Kitchen
Quality`, and `Central Air Conditioning`.

**Step 2: Add category display mappings and fallback formatting**

Define readable mappings for House Prices neighborhood codes, kitchen-quality
codes, and central-air codes. Implement a helper that returns the mapped label
for a raw category and a safe title-cased fallback for an unseen category.

**Step 3: Wire the mappings into the page**

Pass the explicit feature label to each numeric input. Pass the explicit
categorical label and a `format_func` closure to each selectbox. Do not change
the `key`, options list, or prediction-service payload shape.

**Step 4: Run the focused test to verify it passes**

Run: `pytest tests/test_house_app.py -q`

Expected: PASS with all House Pricing page tests green.

### Task 3: Verify the complete change

**Files:**
- Inspect: `app.py`, `tests/test_house_app.py`, and the two plan documents.

**Step 1: Run the full test suite**

Run: `pytest -q`

Expected: PASS with zero failures.

**Step 2: Inspect the diff and worktree**

Run: `git diff HEAD^ -- app.py tests/test_house_app.py docs/plans/2026-09-21-house-ui-human-friendly-inputs-design.md docs/plans/2026-09-21-house-ui-human-friendly-inputs.md` and `git status --short`.

Expected: only the approved presentation code, focused tests, and plan
documents are changed; the pre-existing AWS CodePipeline plan remains
untracked and untouched.

**Step 3: Commit the implementation**

Run: `git add app.py tests/test_house_app.py docs/plans/2026-09-21-house-ui-human-friendly-inputs.md && git commit -m "feat: humanize house pricing inputs"`
