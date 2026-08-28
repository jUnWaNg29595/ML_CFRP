"""Compatibility package for the vendored BigSMILES UI renderer."""

from pathlib import Path

# Keep both historical import forms working:
# ``bigsmiles_ui.renderer`` and ``bigsmiles_ui.bigsmiles_ui.renderer``.
_nested = str(Path(__file__).with_name("bigsmiles_ui"))
if _nested not in __path__:
    __path__.append(_nested)

