from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import project_path
from src.label_tool.kline_loader import KlineRequest, load_kline


CANDIDATE_COLUMNS = [
    "candidate_id",
    "start_date",
    "end_date",
    "region_type",
    "cycle",
    "level",
    "label_freq",
    "suggested_entry_start",
    "suggested_entry_end",
    "suggested_exit_start",
    "suggested_exit_end",
    "evidence",
    "score",
    "notes",
]

DEFAULT_CANDIDATE_PATH = project_path("data", "manual_labels", "turning_region_candidates.csv")


@dataclass(frozen=True)
class CandidateSpec:
    cycle: str
    level: str
    label_freq: str
    pivot_window: int
    region_radius: int
    future_window: int
    min_move: float
    max_rows: int


SPECS = (
    CandidateSpec("short", "swing", "daily", 5, 4, 15, 0.045, 18),
    CandidateSpec("medium", "swing", "weekly", 15, 10, 40, 0.09, 14),
    CandidateSpec("long", "major", "monthly", 45, 22, 120, 0.16, 10),
)


def _fmt_date(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _clip_date(df: pd.DataFrame, pos: int) -> str:
    pos = int(max(0, min(len(df) - 1, pos)))
    return _fmt_date(df.iloc[pos]["trade_date"])


def _candidate_id(region_type: str, cycle: str, pivot_date: str, seq: int) -> str:
    prefix = "B" if region_type == "bottom" else "T"
    return f"C_{prefix}{pivot_date[:7].replace('-', '')}_{cycle}_{seq:02d}"


def _score_bottom(row: pd.Series, spec: CandidateSpec) -> float:
    future = float(row["future_up"])
    prior = float(abs(row["prior_down"]))
    rsi_bonus = max(0.0, (35.0 - float(row.get("rsi6") or 50.0)) / 35.0)
    boll_bonus = 0.25 if bool(row.get("below_boll")) else 0.0
    return future * 5.0 + prior * 2.0 + rsi_bonus + boll_bonus


def _score_top(row: pd.Series, spec: CandidateSpec) -> float:
    future = float(abs(row["future_down"]))
    prior = float(row["prior_up"])
    rsi_bonus = max(0.0, (float(row.get("rsi6") or 50.0) - 65.0) / 35.0)
    boll_bonus = 0.25 if bool(row.get("above_boll")) else 0.0
    return future * 5.0 + prior * 2.0 + rsi_bonus + boll_bonus


def _build_evidence(row: pd.Series, region_type: str, spec: CandidateSpec) -> str:
    parts = [
        f"pivot={_fmt_date(row['trade_date'])}",
        f"future_window={spec.future_window}d",
    ]
    if region_type == "bottom":
        parts.extend(
            [
                f"future_max_ret={float(row['future_up']):.2%}",
                f"prior_drawdown={float(row['prior_down']):.2%}",
            ]
        )
        if bool(row.get("below_boll")):
            parts.append("below_boll_2")
        if pd.notna(row.get("rsi6")):
            parts.append(f"rsi6={float(row['rsi6']):.1f}")
    else:
        parts.extend(
            [
                f"future_min_ret={float(row['future_down']):.2%}",
                f"prior_rise={float(row['prior_up']):.2%}",
            ]
        )
        if bool(row.get("above_boll")):
            parts.append("above_boll_2")
        if pd.notna(row.get("rsi6")):
            parts.append(f"rsi6={float(row['rsi6']):.1f}")
    return "|".join(parts)


def _scan_one(df: pd.DataFrame, spec: CandidateSpec, region_type: str) -> list[dict]:
    data = df.copy().reset_index(drop=True)
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    win = spec.pivot_window
    fw = spec.future_window

    data["roll_low"] = low.rolling(win * 2 + 1, center=True, min_periods=win + 1).min()
    data["roll_high"] = high.rolling(win * 2 + 1, center=True, min_periods=win + 1).max()
    data["future_max"] = close.shift(-1).rolling(fw, min_periods=max(3, fw // 4)).max().shift(-(fw - 1))
    data["future_min"] = close.shift(-1).rolling(fw, min_periods=max(3, fw // 4)).min().shift(-(fw - 1))
    data["prior_max"] = close.rolling(fw, min_periods=max(3, fw // 4)).max()
    data["prior_min"] = close.rolling(fw, min_periods=max(3, fw // 4)).min()
    data["future_up"] = data["future_max"] / close - 1
    data["future_down"] = data["future_min"] / close - 1
    data["prior_down"] = close / data["prior_max"] - 1
    data["prior_up"] = close / data["prior_min"] - 1
    data["below_boll"] = close <= pd.to_numeric(data.get("boll_lower_2"), errors="coerce")
    data["above_boll"] = close >= pd.to_numeric(data.get("boll_upper_2"), errors="coerce")

    if region_type == "bottom":
        part = data[(low <= data["roll_low"]) & (data["future_up"] >= spec.min_move)].copy()
        part["score"] = part.apply(lambda row: _score_bottom(row, spec), axis=1)
        part = part.sort_values("score", ascending=False)
    else:
        part = data[(high >= data["roll_high"]) & (data["future_down"] <= -spec.min_move)].copy()
        part["score"] = part.apply(lambda row: _score_top(row, spec), axis=1)
        part = part.sort_values("score", ascending=False)
    scores = part["score"].to_dict()

    chosen: list[int] = []
    min_gap = max(5, spec.region_radius * 2)
    for idx in part.index.tolist():
        if all(abs(idx - existing) >= min_gap for existing in chosen):
            chosen.append(idx)
        if len(chosen) >= spec.max_rows:
            break

    records: list[dict] = []
    for seq, idx in enumerate(sorted(chosen, key=lambda i: data.loc[i, "trade_date"]), start=1):
        row = data.loc[idx]
        start = _clip_date(data, idx - spec.region_radius)
        end = _clip_date(data, idx + spec.region_radius)
        pivot = _fmt_date(row["trade_date"])
        candidate = {
            "candidate_id": _candidate_id(region_type, spec.cycle, pivot, seq),
            "start_date": start,
            "end_date": end,
            "region_type": region_type,
            "cycle": spec.cycle,
            "level": spec.level,
            "label_freq": spec.label_freq,
            "suggested_entry_start": "",
            "suggested_entry_end": "",
            "suggested_exit_start": "",
            "suggested_exit_end": "",
            "evidence": _build_evidence(row, region_type, spec),
            "score": f"{float(scores.get(idx, 0.0)):.4f}",
            "notes": "大模型辅助候选，需人工确认",
        }
        if region_type == "bottom":
            candidate["suggested_entry_start"] = pivot
            candidate["suggested_entry_end"] = end
        else:
            candidate["suggested_exit_start"] = start
            candidate["suggested_exit_end"] = pivot
        records.append(candidate)
    return records


def generate_candidates() -> pd.DataFrame:
    daily = load_kline(KlineRequest(freq="daily"))
    if daily.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    records: list[dict] = []
    for spec in SPECS:
        records.extend(_scan_one(daily, spec, "bottom"))
        records.extend(_scan_one(daily, spec, "top"))
    out = pd.DataFrame(records)
    if out.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    out = out[CANDIDATE_COLUMNS].sort_values(["start_date", "region_type", "cycle"]).reset_index(drop=True)
    return out


class CandidateStore:
    def __init__(self, path: str | Path = DEFAULT_CANDIDATE_PATH):
        self.path = Path(path)

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)
        df = pd.read_csv(self.path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        for col in CANDIDATE_COLUMNS:
            if col not in df:
                df[col] = ""
        return self._sort(df[CANDIDATE_COLUMNS].fillna(""))

    @staticmethod
    def _sort(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "start_date" not in df:
            return df.reset_index(drop=True)
        out = df.copy()
        out["_sort_start"] = pd.to_datetime(out["start_date"], errors="coerce")
        out["_sort_end"] = pd.to_datetime(out.get("end_date", ""), errors="coerce")
        out = out.sort_values(
            ["_sort_start", "_sort_end", "region_type", "cycle", "label_freq", "candidate_id"],
            na_position="last",
            kind="mergesort",
        )
        return out.drop(columns=["_sort_start", "_sort_end"]).reset_index(drop=True)

    def write(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sort(df[CANDIDATE_COLUMNS].fillna("")).to_csv(self.path, index=False, encoding="utf-8-sig")

    def regenerate(self) -> list[dict]:
        df = generate_candidates()
        self.write(df)
        return self.list_records()

    def list_records(self) -> list[dict]:
        return self.read().to_dict(orient="records")

    def delete(self, candidate_id: str) -> None:
        df = self.read()
        if candidate_id not in set(df["candidate_id"].tolist()):
            raise KeyError(f"候选不存在: {candidate_id}")
        self.write(df[df["candidate_id"] != candidate_id].copy())
