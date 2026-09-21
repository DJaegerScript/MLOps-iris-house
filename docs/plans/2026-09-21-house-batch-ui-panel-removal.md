# House Pricing Batch UI Panel Removal Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Hide the House Pricing page's visible Batch prediction panel without changing batch prediction backend behavior.

**Architecture:** Remove only the batch UI render block from `_render_house_page` in `app.py`. Keep `HouseBatchPredictionService`, `HouseRuntime.batch_service`, validation, metrics, drift calculations, and their tests unchanged. Extend the page-boundary test to prove the heading and uploader are not emitted while preserving single-property prediction coverage.

**Tech Stack:** Python 3.11, Streamlit, pytest, git.

---

### Task 1: Add the failing UI regression assertion

**Files:**
- Modify: `tests/test_house_app.py:104-159`

**Step 1: Update the House page test**

In `test_house_page_renders_all_declared_inputs_versions_and_prediction`, keep
the existing input and single-property prediction assertions and add checks
that no recorded call has:

- `subheader` value equal to `"Batch prediction"`
- `file_uploader` call

Do not remove the fake uploader implementation yet; retaining it lets the test
observe an accidental render call without coupling the fixture to an
`AttributeError`.

**Step 2: Run the focused test to verify it fails**

Run:

```bash
pytest tests/test_house_app.py::test_house_page_renders_all_declared_inputs_versions_and_prediction -q
```

Expected: FAIL because the current House page records the Batch prediction
subheader and file uploader calls.

### Task 2: Remove the visible batch render block

**Files:**
- Modify: `app.py:225-270`

**Step 1: Remove the page render block**

Delete the `_render_house_page` section beginning with the optional
`st.file_uploader` lookup and ending after the batch drift warning. This removes
the heading, uploader, batch action, result table, download button, metrics,
drift diagnostics, and row summary from the House page.

Leave `HouseRuntime.batch_service` construction and all code in
`house_pricing_mlops.prediction` unchanged so batch functionality remains
available for future surfaces and its existing tests continue to apply.

**Step 2: Run the focused House app tests**

Run:

```bash
pytest tests/test_house_app.py -q
```

Expected: PASS with no House page regressions.

### Task 3: Verify the scoped change and repository behavior

**Files:**
- No additional source files.

**Step 1: Check the diff for whitespace errors**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

**Step 2: Run the full test suite**

Run:

```bash
pytest -q
```

Expected: the full suite exits with code 0. If an environment or dependency
blocker occurs, report it separately from the focused test result.

**Step 3: Review the final diff**

Run:

```bash
git status --short
git diff -- app.py tests/test_house_app.py docs/plans/2026-09-21-house-batch-ui-panel-removal.md
```

Confirm the only implementation change is removal of the visible batch UI and
the only test change asserts that the batch heading and uploader are absent.

**Step 4: Commit the implementation**

```bash
git add app.py tests/test_house_app.py docs/plans/2026-09-21-house-batch-ui-panel-removal.md
git commit -m "fix: hide house batch prediction panel"
```
