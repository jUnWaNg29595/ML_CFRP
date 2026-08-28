# Training Context Fix Report

## Change

Added `select_training_context_for_model` to keep the approved numeric feature
contract strict for ordinary models while skipping it for graph and raw-frame
model input contracts. The training page now uses the selected model context for
single-split training and optional cross-validation; the original context remains
used for audit and artifact persistence.

## Verification

Command:

```text
python -m pytest -q tests/test_training_contract.py::test_model_specific_training_inputs_do_not_reuse_strict_numeric_context
```

Output:

```text
.                                                                        [100%]
1 passed in 0.75s
```

Command:

```text
python -m compileall -q app.py core/training_contract.py core/model_trainer.py
```

Output:

```text
(no output; exit code 0)
```

The full `tests/test_training_contract.py` run reached 3 passed tests but hit an
existing environment incompatibility while importing SciPy:
`AttributeError: module 'numpy' has no attribute 'long'`.

## Scope note

`app.py` already contained unrelated concurrent edits, so only
`core/training_contract.py` and `tests/test_training_contract.py` are included in
the focused commit. The app call-site changes remain in the shared working tree
for the parent task's final integration commit.

## Follow-up review fix

The training page now imports `RAW_FRAME_MODEL_NAMES` from
`core.model_trainer` and defines `raw_frame_models` as that complete set plus
`Epoxy PINN (Physics-Informed)`. A source-level regression test verifies this
coverage and the model-specific context routing.

Follow-up verification:

```text
python -m pytest -q tests/test_training_contract.py::test_training_page_routes_model_specific_context_for_all_raw_frame_models tests/test_training_contract.py::test_model_specific_training_inputs_do_not_reuse_strict_numeric_context
..                                                                       [100%]
2 passed in 0.75s

python -m compileall -q app.py core/training_contract.py core/model_trainer.py
(no output; exit code 0)
```
