from __future__ import annotations

import csv
from pathlib import Path

from src.macro_events.cleaner import clean_macro_events
from src.macro_events.schemas import MacroEvent


def load_events_csv(path: str | Path) -> list[MacroEvent]:
    csv_path = Path(path)
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with csv_path.open("r", encoding=encoding, newline="") as file:
                rows = list(csv.DictReader(file))
            return clean_macro_events(MacroEvent.from_dict(row) for row in rows)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"无法识别 CSV 编码: {csv_path}")
