# Final Review Fix Report

## Scope

- `core/prediction_portal.py`: validate approved dataset manifests against the full registry payload (or full registry), bind contract/profile IDs, and require canonical `workflow_source_fields` for v2 contracts while retaining aliases for schema-1/legacy compatibility.
- `core/feature_mapping_review.py`: require an explicit `status=approved` on suggestions accepted or edited into approved bindings.
- `core/feature_registry_ui.py`: expose `unknown` suggestions in the status filter.
- Added regression tests covering manifest tampering/approval, profile binding, v2 source-field aliases, missing review status, and the UI filter.

## Verification

Command:

`C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_prediction_portal.py tests/test_portal_prediction.py tests/test_feature_mapping_review.py tests/test_feature_registry_ui.py tests/test_feature_registry_end_to_end.py tests/test_legacy_tg_gate.py -q`

Result: `72 passed`.

Command:

`C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m compileall -q core tests`

Result: passed.

## Whole-Branch Training Context Fix

Commits `ecbf3d1` and `51081b4` add model-specific training context routing. Graph and raw-frame models use their own input columns without applying the strict numeric contract; ordinary models retain strict canonical-column validation. The app now uses the complete `RAW_FRAME_MODEL_NAMES` set plus the Epoxy PINN path for both training and cross-validation.

Coverage:

`C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest -q tests/test_training_contract.py::test_training_page_routes_model_specific_context_for_all_raw_frame_models tests/test_training_contract.py::test_model_specific_training_inputs_do_not_reuse_strict_numeric_context`

Output: `2 passed`.

`C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m compileall -q app.py core/training_contract.py core/model_trainer.py`

Output: passed (no output).
