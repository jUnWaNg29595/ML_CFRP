# app task lock fix report

STATUS: complete

## Changes

- Added `active_task_lock = bool(get_task_manager().get_active_tasks())` at the start of `_render_network_proxy_panel` in `app.py`, before any widget using `disabled=active_task_lock`.
- No unrelated files were changed.

## Verification

Command:

`C:/Users/wangj/anaconda3/envs/CFRP_env/python.exe -m pytest tests/test_task_manager.py -q`

Output:

`7 passed in 0.36s`

## Commit

`82a3a2d48bff789f2697fc20d5d7f69534f18e2e`

## Concerns

The function parameter is intentionally refreshed from the task manager at panel render time so the lock reflects current active tasks; existing callers and widget behavior remain unchanged.
