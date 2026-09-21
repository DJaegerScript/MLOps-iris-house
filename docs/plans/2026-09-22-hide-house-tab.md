# Hide House Pricing Tab in Iris Streamlit Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove House Pricing from the Iris Streamlit product navigation without changing the standalone House runtime.

**Architecture:** Keep the existing `APP_VARIANT` split. In the Iris branch, constrain the existing product radio to `IRIS_PRODUCT`; leave the House branch and `_render_house_page()` unchanged.

**Tech Stack:** Python, Streamlit, pytest.

---

### Task 1: Add the navigation regression test

**Files:**
- Modify: `tests/test_house_app.py`
- Test: `tests/test_house_app.py`

**Step 1: Write the failing test**

Extend the Iris navigation test fixture so it records the radio options, then assert the Iris product navigation offers exactly `[app.IRIS_PRODUCT]`.

**Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_house_app.py -q`

Expected: FAIL because the current navigation still offers `HOUSE_PRODUCT`.

### Task 2: Restrict Iris navigation

**Files:**
- Modify: `app.py:_selected_product`

**Step 1: Write the minimal implementation**

Change the existing product radio options from `[IRIS_PRODUCT, HOUSE_PRODUCT]` to `[IRIS_PRODUCT]`. Do not remove House imports, page rendering, or the `APP_VARIANT=house` branch.

**Step 2: Run the focused tests**

Run: `python3 -m pytest tests/test_app.py tests/test_house_app.py -q`

Expected: PASS with the Iris and House Streamlit boundary tests green.

### Task 3: Verify and publish

**Files:**
- Inspect: `app.py`, `tests/test_house_app.py`

**Step 1: Run the complete verification**

Run: `python3 -m pytest -q`

Expected: PASS with zero failures.

**Step 2: Check the patch**

Run: `git diff --check` and inspect `git diff --cached` to confirm only the intended implementation, test, and plan changes are staged; preserve the unrelated untracked AWS plan.

**Step 3: Commit and push**

Run: `git add app.py tests/test_house_app.py docs/plans/2026-09-22-hide-house-tab.md` and `git commit -m "fix: hide House tab from Iris app"`, then `git push origin main`.

**Step 4: Verify remote delivery**

Run: `git status --short --branch` and `git ls-remote origin refs/heads/main` and confirm the pushed `main` ref matches the local `HEAD` commit.
