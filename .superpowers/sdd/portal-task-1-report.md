# Task 1 Report

Status: complete

Commit hash: 76c28263e8ebaff6af20e6bb822e796df2875a0a

## Tests

- `& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_prediction_portal.py' -q` (initial RED: module not found; after implementation: 8 passed, 1 failed due to test-message wording, then fixed)
- `& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_prediction_portal.py' 'tests/test_prediction_feature_contract.py' 'tests/test_prediction_molecular_baseline.py' -q` (16 passed)
- `git diff --check` (passed)

## Changes

- Added strict prediction contract construction and publication validation.
- Recorded exact feature order, target, workflow metadata, source columns, preprocessing presence, and finite numeric ranges.
- Added versioned publication entry creation, single-active-version activation, rollback, health labels, gating, and active-release selection.
- Added focused tests for contract, workflow rejection, legacy artifacts, version behavior, health labels, and publication gating.

## Concerns

- Main-platform export wiring and external portal integration remain for later plan tasks.

## Review Fixes

- Included the exact existing `core/prediction_contract.py` and `core/prediction_molecular_baseline.py` helper files so this branch is self-contained.
- Required the complete prediction contract schema, finite numeric ranges, workflow source consistency, workflow hash/schema consistency, and usable fitted preprocessor metadata.
- Blocked molecular feature artifacts without a saved reproducible workflow.
- Validated rollback versions before mutating enabled flags; unknown versions leave the active release unchanged.
- Rejected multiple enabled releases instead of silently selecting the first one.
- Reconciled pipeline, imputer, and scaler presence flags for every artifact type.
- Rejected explicit contracts that differ from a saved artifact contract.
- Rejected preprocessors with empty or None-valued learned state and pipelines with placeholder steps.
- Rejected duplicate rollback versions before mutating configuration.

## Review Fix Tests

- `& 'C:\Users\wangj\anaconda3\envs\CFRP_env\python.exe' -m pytest 'tests/test_prediction_portal.py' 'tests/test_prediction_feature_contract.py' 'tests/test_prediction_molecular_baseline.py' -q`
  - Output: `............................                                             [100%]` / `28 passed in 0.33s`
- `git diff --check` passed.
