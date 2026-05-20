from __future__ import annotations

import sys


def disable_broken_dask_autoload() -> None:
    """Let LightGBM import without loading an incompatible local dask build."""
    sys.modules.setdefault("dask", None)
    sys.modules.setdefault("dask.dataframe", None)

