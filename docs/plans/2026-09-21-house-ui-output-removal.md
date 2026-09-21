# House Pricing UI Output Removal Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the House Pricing page's educational warning and test-metric display without changing model evaluation or batch behavior.

**Architecture:** Keep all metric calculation and persistence in the House Pricing backend unchanged. Narrow the Streamlit change to the page render function by removing the warning output and the helper that formats test metrics; update the page-boundary test to assert those outputs are absent.

**Tech Stack:** Python 3.11, Streamlit, pytest, git.

---

### Task 1: Add the regression assertion

**Files:**
- Modify: `tests/test_house_app.py`

**Step 1: Write the failing test**

Update the House Pricing page test fixture to provide representative test
metrics and assert that the page emits neither the educational warning nor the
test metric mapping. Remove the existing assertion that requires the warning.

**Step 2: Run the focused test to verify it fails**

Run:

```bash
pytest tests/test_house_app.py::test_house_page_renders_all_declared_inputs_versions_and_prediction -q
```

Expected: FAIL because the current page still emits the warning and test
metrics.

### Task 2: Remove the two UI outputs

**Files:**
- Modify: `app.py:225-335`

**Step 1: Remove the warning render**

Delete the `st.info` call containing the educational warning from
`_render_house_page`.

**Step 2: Remove the test-metric render helper and call**

Delete the `_render_house_metrics` call and the now-unused helper. Do not alter
metric calculation, model loading, batch metrics, or drift diagnostics.

**Step 3: Run the focused test to verify it passes**

Run:

```bash
pytest tests/test_house_app.py -q
```

Expected: PASS with no House app test failures.

### Task 3: Run repository verification

**Files:**
- No additional source files.

**Step 1: Check formatting and test coverage**

Run:

```bash
git diff --check
pytest -q
```

Expected: `git diff --check` is silent and the full test suite exits with code
0.

**Step 2: Review the final diff**

Run:

```bash
git status --short
git diff -- app.py tests/test_house_app.py
```

Confirm only the requested UI outputs and their test expectations changed.

### Task 4: Commit and push

**Files:**
- `app.py`
- `tests/test_house_app.py`
- `docs/plans/2026-09-21-house-ui-output-removal.md`

**Step 1: Commit the implementation**

```bash
git add app.py tests/test_house_app.py docs/plans/2026-09-21-house-ui-output-removal.md
git commit -m "fix: remove house UI warning and test metrics"
```

**Step 2: Verify the branch and remote target**

```bash
git status --short --branch
git branch --show-current
git remote get-url origin
```

Expected: current branch is `main`, the working tree is clean, and `origin`
points to the project repository.

**Step 3: Push to main**

```bash
git push origin main
```

Expected: the remote `main` ref advances successfully.
