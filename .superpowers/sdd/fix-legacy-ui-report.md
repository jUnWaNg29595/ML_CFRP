# Legacy UI compatibility fix

## Scope

- Added molecular-feature column collection and session cleanup with named snapshot backup.
- Added clear/restore controls to molecular feature and feature selection pages while preserving raw `data`.
- Kept canonical SMILES / BigSMILES navigation name and exposed legacy structure-recognition entrypoint.
- Added compatibility package path for both `bigsmiles_ui.renderer` and `bigsmiles_ui.bigsmiles_ui.renderer` imports.
- Restored training-record overview and optimization-detail labels required by the UI contract.

## Verification

Command:

`C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_app_scope_regressions.py tests/test_smiles_structure_ui_contract.py tests/test_virtual_screening.py -q`

Result: **73 passed**.

The requested `tests/test_structure_visualization_ui.py` file is not present in this checkout, so it could not be included in the command. Both renderer import paths were additionally exercised directly; each rendered `CCO` with `valid rendered` status and produced an image file.
