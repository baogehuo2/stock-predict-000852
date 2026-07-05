from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from src.common.config import project_path
from src.label_tool.candidates import CandidateStore, DEFAULT_CANDIDATE_PATH
from src.label_tool.kline_loader import KlineRequest, kline_to_records, load_kline, set_target_index
from src.label_tool.label_store import DEFAULT_LABEL_PATH, LabelStore


STATIC_DIR = Path(__file__).resolve().parent / "static"


class LabelToolHandler(BaseHTTPRequestHandler):
    store = LabelStore()
    candidate_store = CandidateStore()
    index_code = "000852"
    index_name = "中证1000"
    default_freq = "daily"
    allowed_freqs = {"daily", "weekly", "monthly", "intraday"}
    enable_candidates = True

    def log_message(self, fmt: str, *args: object) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"ok": False, "error": message}, status)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _serve_static(self, path: str) -> None:
        target = STATIC_DIR / ("index.html" if path in ("", "/") else path.lstrip("/"))
        if not target.resolve().is_relative_to(STATIC_DIR.resolve()) or not target.exists() or target.is_dir():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix in {".html", ".css", ".js"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        if target.suffix in {".html", ".css", ".js"}:
            self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/kline":
            qs = parse_qs(parsed.query)
            try:
                req = KlineRequest(
                    freq=qs.get("freq", ["daily"])[0],
                    start_date=qs.get("start_date", [None])[0] or None,
                    end_date=qs.get("end_date", [None])[0] or None,
                )
                if req.freq not in self.allowed_freqs:
                    raise ValueError(f"当前工具只允许查看: {', '.join(sorted(self.allowed_freqs))}")
                data = kline_to_records(load_kline(req))
                self._send_json({"ok": True, "data": data})
            except Exception as exc:
                self._send_error(str(exc), 500)
            return
        if parsed.path == "/api/meta":
            self._send_json(
                {
                    "ok": True,
                    "index_code": self.index_code,
                    "index_name": self.index_name,
                    "default_freq": self.default_freq,
                    "allowed_freqs": sorted(self.allowed_freqs),
                    "enable_candidates": self.enable_candidates,
                    "label_path": str(self.store.path),
                    "candidate_path": str(self.candidate_store.path),
                }
            )
            return
        if parsed.path == "/api/labels":
            self._send_json({"ok": True, "data": self.store.list_records(), "path": str(self.store.path)})
            return
        if parsed.path == "/api/candidates":
            if not self.enable_candidates:
                self._send_json({"ok": True, "data": [], "path": str(self.candidate_store.path)})
                return
            self._send_json({"ok": True, "data": self.candidate_store.list_records(), "path": str(self.candidate_store.path)})
            return
        if parsed.path == "/api/export":
            self._send_json({"ok": True, "path": str(self.store.path), "data": self.store.list_records()})
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/labels":
            if parsed.path == "/api/candidates/regenerate":
                if not self.enable_candidates:
                    self._send_error("当前周线工具禁用日线候选扫描。", 400)
                    return
                try:
                    items = self.candidate_store.regenerate()
                    self._send_json({"ok": True, "data": items, "path": str(self.candidate_store.path)})
                except Exception as exc:
                    self._send_error(str(exc), 500)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            item = self.store.add(self._read_json())
            self._send_json({"ok": True, "data": item})
        except Exception as exc:
            self._send_error(str(exc), 400)

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/labels/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        region_id = unquote(parsed.path.removeprefix("/api/labels/"))
        try:
            item = self.store.update(region_id, self._read_json())
            self._send_json({"ok": True, "data": item})
        except KeyError as exc:
            self._send_error(str(exc), 404)
        except Exception as exc:
            self._send_error(str(exc), 400)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/labels/"):
            region_id = unquote(parsed.path.removeprefix("/api/labels/"))
            try:
                self.store.delete(region_id)
                self._send_json({"ok": True})
            except KeyError as exc:
                self._send_error(str(exc), 404)
            except Exception as exc:
                self._send_error(str(exc), 400)
            return
        if parsed.path.startswith("/api/candidates/"):
            candidate_id = unquote(parsed.path.removeprefix("/api/candidates/"))
            try:
                self.candidate_store.delete(candidate_id)
                self._send_json({"ok": True})
            except KeyError as exc:
                self._send_error(str(exc), 404)
            except Exception as exc:
                self._send_error(str(exc), 400)
            return
        self.send_error(HTTPStatus.NOT_FOUND)


def run(
    host: str = "127.0.0.1",
    port: int = 8765,
    label_path: str | None = None,
    index_code: str | None = None,
    index_name: str | None = None,
    default_freq: str = "daily",
    allowed_freqs: str | None = None,
    enable_candidates: bool = True,
) -> None:
    selected_index = index_code or "000852"
    set_target_index(selected_index)
    LabelToolHandler.index_code = selected_index
    LabelToolHandler.index_name = index_name or ("沪深300" if selected_index == "000300" else "中证1000")
    LabelToolHandler.default_freq = default_freq
    parsed_allowed = {x.strip() for x in (allowed_freqs or "daily,weekly,monthly,intraday").split(",") if x.strip()}
    LabelToolHandler.allowed_freqs = parsed_allowed or {"daily", "weekly", "monthly", "intraday"}
    LabelToolHandler.enable_candidates = enable_candidates
    if label_path:
        LabelToolHandler.store = LabelStore(project_path(label_path) if not Path(label_path).is_absolute() else label_path)
    else:
        LabelToolHandler.store = LabelStore(DEFAULT_LABEL_PATH)
    LabelToolHandler.candidate_store = CandidateStore(DEFAULT_CANDIDATE_PATH)
    server = ThreadingHTTPServer((host, port), LabelToolHandler)
    print(f"顶底标注工具已启动: http://{host}:{port}")
    print(f"指数: {LabelToolHandler.index_name} {LabelToolHandler.index_code}")
    print(f"默认频率: {LabelToolHandler.default_freq}; 允许频率: {','.join(sorted(LabelToolHandler.allowed_freqs))}")
    print(f"标注文件: {LabelToolHandler.store.path}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run turning region label tool.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--label-path", default=None)
    parser.add_argument("--index-code", default=None)
    parser.add_argument("--index-name", default=None)
    parser.add_argument("--default-freq", default="daily")
    parser.add_argument("--allowed-freqs", default=None)
    parser.add_argument("--disable-candidates", action="store_true")
    args = parser.parse_args()
    run(
        args.host,
        args.port,
        args.label_path,
        args.index_code,
        args.index_name,
        args.default_freq,
        args.allowed_freqs,
        not args.disable_candidates,
    )


if __name__ == "__main__":
    main()
