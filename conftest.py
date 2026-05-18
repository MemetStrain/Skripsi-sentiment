"""pytest bootstrap — make the `prediction` package importable in tests.

`prediction/` has no `__init__.py` (it works as an implicit namespace package).
Putting the repo root and `prediction/` on `sys.path` lets the test suite use
`from prediction.master_features import ...` while the prediction modules keep
their own `from master_features import ...` / `from utils.forecast_utils import
...` style imports.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "prediction")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
