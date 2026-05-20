from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import get_config, load_yaml, project_path
from src.common.db import get_engine
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.common.network import disable_env_proxies
from src.common.secrets import load_secrets


def check(name: str, func) -> bool:
    try:
        func()
        print(f"[OK] {name}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        return False


def main() -> int:
    disable_env_proxies()
    results = []
    results.append(check("config/config.yaml 可读取", lambda: get_config()))
    results.append(check("config/db.yaml 可读取", lambda: load_yaml(project_path("config", "db.yaml"))))
    results.append(check("加密配置可解密", lambda: load_secrets()))
    results.append(check("核心目录存在", lambda: [project_path(p).mkdir(parents=True, exist_ok=True) for p in ["logs", "models", "data/reports", "data/raw", "data/processed"]]))
    results.append(check("数据库可连接", lambda: get_engine().connect().close()))
    def _import_core() -> None:
        disable_broken_dask_autoload()
        for module in ["pandas", "numpy", "sqlalchemy", "sklearn", "lightgbm"]:
            importlib.import_module(module)

    results.append(check("pandas/numpy/sqlalchemy/sklearn/lightgbm 可导入", _import_core))

    def _akshare_probe() -> None:
        from src.collectors.collect_index_akshare import fetch_index_history

        df = fetch_index_history("000852", "中证1000", "2024-01-01", "2024-01-31")
        if df is None or df.empty:
            raise RuntimeError("AKShare 返回空数据")

    results.append(check("AKShare 中证1000行情探测", _akshare_probe))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
