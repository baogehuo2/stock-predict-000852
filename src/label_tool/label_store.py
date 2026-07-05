from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import project_path
from src.label_tool.validators import CSV_COLUMNS, validate_label


DEFAULT_LABEL_PATH = project_path("data", "manual_labels", "market_turning_regions.csv")


class LabelStore:
    def __init__(self, path: str | Path = DEFAULT_LABEL_PATH):
        self.path = Path(path)

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=CSV_COLUMNS)
        df = pd.read_csv(self.path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return self._sort(df[CSV_COLUMNS].fillna(""))

    @staticmethod
    def _sort(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "start_date" not in df:
            return df.reset_index(drop=True)
        out = df.copy()
        out["_sort_start"] = pd.to_datetime(out["start_date"], errors="coerce")
        out["_sort_end"] = pd.to_datetime(out.get("end_date", ""), errors="coerce")
        out = out.sort_values(
            ["_sort_start", "_sort_end", "region_type", "cycle", "label_freq", "region_id"],
            na_position="last",
            kind="mergesort",
        )
        return out.drop(columns=["_sort_start", "_sort_end"]).reset_index(drop=True)

    def write(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        for col in CSV_COLUMNS:
            if col not in out.columns:
                out[col] = ""
        out = out[CSV_COLUMNS].fillna("")
        out = self._sort(out)
        out.to_csv(self.path, index=False, encoding="utf-8-sig")

    def list_records(self) -> list[dict]:
        return self.read().to_dict(orient="records")

    def add(self, raw: dict) -> dict:
        df = self.read()
        item = validate_label(raw, set(df["region_id"].tolist()))
        df = pd.concat([df, pd.DataFrame([item])], ignore_index=True)
        self.write(df)
        return item

    def update(self, region_id: str, raw: dict) -> dict:
        df = self.read()
        if region_id not in set(df["region_id"].tolist()):
            raise KeyError(f"标注不存在: {region_id}")
        existing_ids = set(df["region_id"].tolist())
        item = validate_label(raw, existing_ids, original_id=region_id)
        df.loc[df["region_id"] == region_id, CSV_COLUMNS] = [item[col] for col in CSV_COLUMNS]
        self.write(df)
        return item

    def delete(self, region_id: str) -> None:
        df = self.read()
        if region_id not in set(df["region_id"].tolist()):
            raise KeyError(f"标注不存在: {region_id}")
        self.write(df[df["region_id"] != region_id].copy())
