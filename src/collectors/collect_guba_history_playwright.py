from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.collect_guba_eastmoney import bar_code_from_url, page_url
from src.common.config import load_yaml, project_path
from src.common.db import init_database, upsert_dataframe
from src.common.logger import get_logger


logger = get_logger(__name__)


def _selected_bars(bar_name: str | None = None) -> list[dict]:
    bars = load_yaml(project_path("config", "symbols.yaml"))["guba_bars"]
    if not bar_name:
        return bars
    return [bar for bar in bars if bar["bar_name"] == bar_name or bar.get("topic") == bar_name]


def _post_id(url: str) -> str:
    match = re.search(r"/news,([^,]+),(\d+)\.html", url)
    return f"{match.group(1)}:{match.group(2)}" if match else url


def _safe_int(value: str | int | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except Exception:
        return None


def _json_field(html: str, name: str) -> str | None:
    match = re.search(rf'"{name}"\s*:\s*"([^"]*)"', html)
    return match.group(1) if match else None


def _json_int(html: str, name: str) -> int | None:
    match = re.search(rf'"{name}"\s*:\s*(\d+)', html)
    return _safe_int(match.group(1)) if match else None


def _is_risk_page(html: str) -> bool:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True).lower()
    risk_keywords = [
        "\u9a8c\u8bc1\u7801",
        "\u5b89\u5168\u9a8c\u8bc1",
        "\u8bbf\u95ee\u8fc7\u4e8e\u9891\u7e41",
        "\u8bf7\u5b8c\u6210\u9a8c\u8bc1",
        "\u6ed1\u5757",
        "\u4eba\u673a\u9a8c\u8bc1",
        "captcha",
        "verify",
        "verification",
    ]
    return any(keyword in text for keyword in risk_keywords)


def _is_forbidden_page(html: str) -> bool:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True).lower()
    return "403 forbidden" in text or "microsoft-iis/10.0" in text


def _canvas_data_url(page, selector: str) -> str | None:
    try:
        return page.evaluate(
            """selector => {
                const node = document.querySelector(selector);
                if (!node || typeof node.toDataURL !== 'function') return null;
                return node.toDataURL('image/png');
            }""",
            selector,
        )
    except Exception:
        return None


def _decode_png(data_url: str):
    import cv2
    import numpy as np

    raw = base64.b64decode(data_url.split(",", 1)[1])
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def _decode_png_bytes(raw: bytes):
    import cv2
    import numpy as np

    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def _decode_data_url_image(data_url: str):
    import cv2
    import numpy as np

    raw = base64.b64decode(data_url.split(",", 1)[1])
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def _data_url_to_bytes(data_url: str | None) -> bytes | None:
    if not data_url or "," not in data_url:
        return None
    try:
        return base64.b64decode(data_url.split(",", 1)[1])
    except Exception:
        return None


def _element_screenshot(page, selector: str) -> bytes | None:
    try:
        locator = page.locator(selector)
        count = locator.count()
        if count == 0:
            return None
        for index in range(count):
            item = locator.nth(index)
            try:
                if item.is_visible(timeout=500):
                    return item.screenshot(timeout=3000)
            except Exception:
                continue
        return locator.first.screenshot(timeout=3000)
    except Exception:
        return None


def _attempt_distance_offset(distance_offset: float | None, attempt: int) -> float:
    attempt_offsets = [0, 15, 25, 35, 50, -10, 65, -20]
    if distance_offset is None:
        return float(attempt_offsets[(attempt - 1) % len(attempt_offsets)])
    return float(distance_offset)


def _safe_debug_name(value: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_")
    return value[:90] or "captcha"


def _write_debug_bytes(debug_dir: Path | None, prefix: str, suffix: str, raw: bytes | None) -> str | None:
    if debug_dir is None or not raw:
        return None
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{prefix}_{suffix}.png"
    path.write_bytes(raw)
    return str(path)


def _write_debug_cv_image(debug_dir: Path | None, prefix: str, suffix: str, image) -> str | None:
    if debug_dir is None or image is None:
        return None
    try:
        import cv2
    except Exception:
        return None
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return None
    return _write_debug_bytes(debug_dir, prefix, suffix, encoded.tobytes())


def _draw_debug_target(debug_dir: Path | None, prefix: str, suffix: str, image, x: float | int | None, box: tuple[int, int, int, int] | None = None) -> None:
    if debug_dir is None or image is None or x is None:
        return
    try:
        import cv2
        import numpy as np
    except Exception:
        return
    if len(image.shape) == 2:
        marked = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        marked = image.copy()
    height = marked.shape[0]
    x_int = int(round(float(x)))
    cv2.line(marked, (x_int, 0), (x_int, height - 1), (0, 0, 255), 2)
    if box:
        bx, by, bw, bh = box
        cv2.rectangle(marked, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
    _write_debug_cv_image(debug_dir, prefix, suffix, marked)


def _write_debug_json(debug_dir: Path | None, prefix: str, data: dict) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{prefix}_meta.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _captcha_debug_prefix(label: str, attempt: int, solver: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"{stamp}_{_safe_debug_name(label)}_attempt{attempt}_{solver}"


def _normalize_captcha_distance(
    raw_distance: float,
    distance_scale: float,
    distance_offset: float | None,
    attempt: int,
    label: str,
    solver: str,
    debug_dir: Path | None = None,
    debug_prefix: str | None = None,
    extra: dict | None = None,
) -> float | None:
    offset = _attempt_distance_offset(distance_offset, attempt)
    distance = raw_distance * distance_scale + offset
    meta = {
        "label": label,
        "solver": solver,
        "attempt": attempt,
        "raw_distance": raw_distance,
        "distance_scale": distance_scale,
        "distance_offset": offset,
        "distance": distance,
        **(extra or {}),
    }
    if raw_distance <= 8:
        meta["accepted"] = False
        meta["reject_reason"] = "too_small"
        if debug_prefix:
            _write_debug_json(debug_dir, debug_prefix, meta)
        logger.warning(
            "captcha distance rejected label=%s solver=%s raw_distance=%.1f reason=too_small",
            label,
            solver,
            raw_distance,
        )
        return None
    if distance < 35 or distance > 260:
        meta["accepted"] = False
        meta["reject_reason"] = "out_of_range"
        if debug_prefix:
            _write_debug_json(debug_dir, debug_prefix, meta)
        logger.warning(
            "captcha distance rejected label=%s solver=%s raw_distance=%.1f distance=%.1f scale=%.3f offset=%s reason=out_of_range",
            label,
            solver,
            raw_distance,
            distance,
            distance_scale,
            offset,
        )
        return None
    meta["accepted"] = True
    if debug_prefix:
        _write_debug_json(debug_dir, debug_prefix, meta)
    return float(distance)


def _eastmoney_piece_pair(page, label: str) -> tuple[bytes | None, bytes | None]:
    piece_raw = _element_screenshot(page, ".em_slice.em_show, .em_slice")
    try:
        page.evaluate(
            """() => {
                document.querySelectorAll('.em_slice').forEach(node => {
                    node.dataset.codexOldVisibility = node.style.visibility || '';
                    node.style.visibility = 'hidden';
                });
            }"""
        )
        bg_raw = _element_screenshot(page, ".em_bg.em_show, .em_bg")
    finally:
        try:
            page.evaluate(
                """() => {
                    document.querySelectorAll('.em_slice').forEach(node => {
                        node.style.visibility = node.dataset.codexOldVisibility || '';
                        delete node.dataset.codexOldVisibility;
                    });
                }"""
            )
        except Exception:
            pass

    if not piece_raw or not bg_raw:
        logger.warning(
            "PuzzleCaptchaSolver screenshots missing label=%s bg=%s piece=%s",
            label,
            bool(bg_raw),
            bool(piece_raw),
        )
    return bg_raw, piece_raw


def _download_eastmoney_captcha_assets(page) -> dict:
    try:
        return page.evaluate(
            """async () => {
                function imageUrl(node) {
                    if (!node) return null;
                    const style = getComputedStyle(node);
                    const candidates = [
                        style.backgroundImage,
                        node.style.backgroundImage,
                        node.getAttribute('src'),
                        node.getAttribute('data-src')
                    ];
                    for (const value of candidates) {
                        if (!value) continue;
                        const match = String(value).match(/url\\(["']?(.*?)["']?\\)/);
                        return match ? match[1] : String(value);
                    }
                    return null;
                }
                function bgUrl(selectors) {
                    for (const selector of selectors) {
                        const url = imageUrl(document.querySelector(selector));
                        if (url) return url;
                    }
                    return null;
                }
                function parsePosition(value) {
                    const parts = String(value || '0px 0px').split(/\\s+/);
                    const x = Number.parseFloat(parts[0] || '0') || 0;
                    const y = Number.parseFloat(parts[1] || '0') || 0;
                    return [x, y];
                }
                async function imageFromUrl(url) {
                    if (!url) return null;
                    const response = await fetch(url, { credentials: 'include' });
                    const blob = await response.blob();
                    const objectUrl = URL.createObjectURL(blob);
                    try {
                        const image = await new Promise((resolve, reject) => {
                            const img = new Image();
                            img.onload = () => resolve(img);
                            img.onerror = reject;
                            img.src = objectUrl;
                        });
                        return image;
                    } finally {
                        URL.revokeObjectURL(objectUrl);
                    }
                }
                async function dataUrl(url) {
                    if (!url) return null;
                    const response = await fetch(url, { credentials: 'include' });
                    const blob = await response.blob();
                    return await new Promise(resolve => {
                        const reader = new FileReader();
                        reader.onloadend = () => resolve(reader.result);
                        reader.readAsDataURL(blob);
                    });
                }
                async function composeSlices(containerSelector, sliceSelector) {
                    const container = document.querySelector(containerSelector);
                    if (!container) return null;
                    const host = container.closest('.em_fullbg, .em_bg') || container;
                    const oldClassName = host.className;
                    const oldStyle = {
                        display: host.style.display,
                        visibility: host.style.visibility,
                        position: host.style.position,
                        left: host.style.left,
                        top: host.style.top,
                        opacity: host.style.opacity
                    };
                    const firstRect = container.getBoundingClientRect();
                    if (!firstRect.width || !firstRect.height) {
                        host.classList.remove('em_hide');
                        host.classList.add('em_show');
                        host.style.display = 'block';
                        host.style.visibility = 'hidden';
                        host.style.position = 'absolute';
                        host.style.left = '-10000px';
                        host.style.top = '0px';
                        host.style.opacity = '1';
                    }
                    const containerRect = container.getBoundingClientRect();
                    const width = Math.round(containerRect.width);
                    const height = Math.round(containerRect.height);
                    if (!width || !height) {
                        host.className = oldClassName;
                        Object.assign(host.style, oldStyle);
                        return null;
                    }
                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');
                    const imageCache = {};
                    const slices = Array.from(container.querySelectorAll(sliceSelector));
                    const sliceMeta = [];
                    for (const slice of slices) {
                        const style = getComputedStyle(slice);
                        const url = imageUrl(slice);
                        if (!url) continue;
                        if (!imageCache[url]) imageCache[url] = await imageFromUrl(url);
                        const img = imageCache[url];
                        if (!img) continue;
                        const rect = slice.getBoundingClientRect();
                        const [posX, posY] = parsePosition(style.backgroundPosition);
                        const sx = Math.max(0, -posX);
                        const sy = Math.max(0, -posY);
                        const sw = Math.round(rect.width);
                        const sh = Math.round(rect.height);
                        const dx = Math.round(rect.left - containerRect.left);
                        const dy = Math.round(rect.top - containerRect.top);
                        ctx.drawImage(img, sx, sy, sw, sh, dx, dy, sw, sh);
                        sliceMeta.push({
                            dx,
                            dy,
                            sw,
                            sh,
                            sx,
                            sy,
                            backgroundPosition: style.backgroundPosition,
                            url
                        });
                    }
                    const output = canvas.toDataURL('image/png');
                    host.className = oldClassName;
                    Object.assign(host.style, oldStyle);
                    return { dataUrl: output, width, height, sliceMeta };
                }
                const bgUrlValue = bgUrl(['.em_cut_bg_slice', '.em_bg', '.em_bg.em_show']);
                const fullUrlValue = bgUrl(['.em_cut_fullbg_slice', '.em_fullbg', '.em_fullbg.em_show']);
                const pieceUrlValue = bgUrl(['.em_slice.em_show', '.em_slice']);
                const bgBox = document.querySelector('.em_bg')?.getBoundingClientRect();
                const pieceBox = document.querySelector('.em_slice.em_show, .em_slice')?.getBoundingClientRect();
                const composedBg = await composeSlices('.em_cut_bg', '.em_cut_bg_slice');
                const composedFull = await composeSlices('.em_cut_fullbg', '.em_cut_fullbg_slice');
                return {
                    bg: composedBg ? composedBg.dataUrl : await dataUrl(bgUrlValue),
                    full: composedFull ? composedFull.dataUrl : await dataUrl(fullUrlValue),
                    piece: await dataUrl(pieceUrlValue),
                    bgUrlValue,
                    fullUrlValue,
                    pieceUrlValue,
                    composed: {
                        bg: !!composedBg,
                        full: !!composedFull
                    },
                    composedBgMeta: composedBg ? { width: composedBg.width, height: composedBg.height, sliceMeta: composedBg.sliceMeta } : null,
                    composedFullMeta: composedFull ? { width: composedFull.width, height: composedFull.height, sliceMeta: composedFull.sliceMeta } : null,
                    displayWidth: bgBox ? bgBox.width : null,
                    bgBox: bgBox ? {
                        left: bgBox.left,
                        top: bgBox.top,
                        width: bgBox.width,
                        height: bgBox.height
                    } : null,
                    pieceBox: pieceBox ? {
                        left: pieceBox.left,
                        top: pieceBox.top,
                        width: pieceBox.width,
                        height: pieceBox.height
                    } : null
                };
            }"""
        )
    except Exception as exc:
        logger.warning("failed to download Eastmoney captcha assets error=%s", exc)
        return {}


def _capture_active_slider_state(
    page,
    handle,
    label: str,
    attempt: int,
    solver: str,
    debug_dir: Path | None,
) -> dict:
    prefix = _captcha_debug_prefix(label, attempt, f"active_{solver}")
    result = {"prefix": prefix, "bg_raw": None, "full_raw": None, "piece_raw": None, "display_width": None}
    handle_box = handle.bounding_box()
    if not handle_box:
        return result

    start_x = handle_box["x"] + handle_box["width"] / 2
    start_y = handle_box["y"] + handle_box["height"] / 2
    active_move = random.uniform(10.0, 18.0)
    try:
        page.mouse.move(start_x, start_y)
        page.mouse.down()
        page.mouse.move(start_x + active_move, start_y + random.uniform(-1.0, 1.0), steps=8)
        page.wait_for_timeout(1000)

        assets = _download_eastmoney_captcha_assets(page)
        bg_raw = _data_url_to_bytes(assets.get("bg"))
        full_raw = _data_url_to_bytes(assets.get("full"))
        piece_raw = _data_url_to_bytes(assets.get("piece"))
        display_width = assets.get("displayWidth")

        if not piece_raw:
            piece_raw = _element_screenshot(page, ".em_slice.em_show, .em_slice")
        if not full_raw:
            full_raw = _element_screenshot(page, ".em_fullbg.em_show, .em_fullbg")
        try:
            page.screenshot(path=str(debug_dir / f"{prefix}_active_page.png"), full_page=True) if debug_dir else None
        except Exception as exc:
            logger.warning("failed to save active captcha screenshot label=%s error=%s", label, exc)

        try:
            page.evaluate(
                """() => {
                    document.querySelectorAll('.em_slice').forEach(node => {
                        node.dataset.codexOldVisibility = node.style.visibility || '';
                        node.style.visibility = 'hidden';
                    });
                }"""
            )
            if not bg_raw:
                bg_raw = _element_screenshot(page, ".em_bg.em_show, .em_bg")
        finally:
            try:
                page.evaluate(
                    """() => {
                        document.querySelectorAll('.em_slice').forEach(node => {
                            node.style.visibility = node.dataset.codexOldVisibility || '';
                            delete node.dataset.codexOldVisibility;
                        });
                    }"""
                )
            except Exception:
                pass

        _write_debug_bytes(debug_dir, prefix, "composed_bg", bg_raw)
        _write_debug_bytes(debug_dir, prefix, "composed_full", full_raw)
        _write_debug_bytes(debug_dir, prefix, "download_piece", piece_raw)
        _write_debug_json(
            debug_dir,
            prefix,
            {
                "label": label,
                "attempt": attempt,
                "solver": solver,
                "active_move": active_move,
                "handle_box": handle_box,
                "asset_urls": {
                    "bg": assets.get("bgUrlValue"),
                    "full": assets.get("fullUrlValue"),
                    "piece": assets.get("pieceUrlValue"),
                },
                "composed": assets.get("composed"),
                "composed_bg_meta": assets.get("composedBgMeta"),
                "composed_full_meta": assets.get("composedFullMeta"),
                "display_width": display_width,
                "bg_box": assets.get("bgBox"),
                "piece_box": assets.get("pieceBox"),
                "captured": {
                    "bg": bool(bg_raw),
                    "full": bool(full_raw),
                    "piece": bool(piece_raw),
                },
            },
        )
        result.update(
            {
                "bg_raw": bg_raw,
                "full_raw": full_raw,
                "piece_raw": piece_raw,
                "display_width": display_width,
                "bg_box": assets.get("bgBox"),
                "piece_box": assets.get("pieceBox"),
            }
        )
        return result
    finally:
        try:
            page.mouse.up()
        except Exception:
            pass
        page.wait_for_timeout(800)

def _eastmoney_image_pair(page) -> tuple[object | None, object | None, float | None]:
    data = _download_eastmoney_captcha_assets(page)
    bg_url = data.get("bgUrlValue")
    full_url = data.get("fullUrlValue")
    if not data.get("bg") or not data.get("full"):
        logger.warning("Eastmoney captcha image urls missing bg_url=%s full_url=%s", bg_url, full_url)
        return None, None, None
    bg = _decode_data_url_image(data["bg"])
    full = _decode_data_url_image(data["full"])
    return bg, full, data.get("displayWidth")


def _find_first_element(page, selectors: list[str]):
    for selector in selectors:
        try:
            node = page.query_selector(selector)
            if node:
                return selector, node
        except Exception:
            continue
    return None, None


def _slider_tracks(distance: float) -> list[float]:
    tracks: list[float] = []
    current = 0.0
    velocity = 0.0
    step_time = 0.18
    mid = distance * 0.62
    while current < distance:
        acceleration = random.uniform(2.0, 4.0) if current < mid else -random.uniform(3.0, 5.5)
        previous_velocity = velocity
        velocity = previous_velocity + acceleration * step_time
        move = previous_velocity * step_time + 0.5 * acceleration * step_time * step_time
        move = max(1.0, move)
        if current + move > distance:
            move = distance - current
        current += move
        tracks.append(move)
    tracks.extend([-2, 2, -1, 1])
    return tracks


def _click_confirm_if_present(page) -> None:
    selectors = ["#btnLogin", ".verify-btn", ".btn-primary", "[class*='confirm']", "[class*='submit']"]
    for selector in selectors:
        try:
            node = page.query_selector(selector)
            if node and node.is_visible():
                node.click(timeout=1000)
                page.wait_for_timeout(800)
                return
        except Exception:
            continue
    for text in ["\u786e\u5b9a", "\u786e\u8ba4", "\u5b8c\u6210", "\u63d0\u4ea4", "\u767b\u5f55"]:
        try:
            locator = page.get_by_text(text, exact=True)
            if locator.count() > 0:
                locator.first.click(timeout=1000)
                page.wait_for_timeout(800)
                return
        except Exception:
            continue


def _reveal_slider_piece(page, handle, label: str) -> bool:
    try:
        handle_box = handle.bounding_box()
        if not handle_box:
            return False
        start_x = handle_box["x"] + handle_box["width"] / 2
        start_y = handle_box["y"] + handle_box["height"] / 2
        trigger_distance = random.randint(55, 85)
        logger.info("trigger slider failure state label=%s distance=%s", label, trigger_distance)
        page.mouse.move(start_x, start_y)
        page.mouse.down()
        current_x = start_x
        for move in _slider_tracks(trigger_distance):
            current_x += move
            page.mouse.move(current_x, start_y + random.uniform(-1.0, 1.0), steps=1)
            page.wait_for_timeout(random.randint(8, 18))
        page.mouse.up()
        page.wait_for_timeout(1800)
        return True
    except Exception as exc:
        logger.warning("failed to trigger slider failure state label=%s error=%s", label, exc)
        return False


def _drag_handle(page, handle, distance: float, label: str) -> bool:
    handle_box = handle.bounding_box()
    if not handle_box:
        return False
    distance = max(20, distance)
    overshoot = random.uniform(8.0, 18.0)
    logger.info("drag slider label=%s distance=%.1f overshoot=%.1f", label, distance, overshoot)
    start_x = handle_box["x"] + handle_box["width"] / 2
    start_y = handle_box["y"] + handle_box["height"] / 2
    page.mouse.move(start_x, start_y)
    page.mouse.down()

    target_x = start_x + distance
    overshoot_x = target_x + overshoot
    current_x = start_x

    forward_steps = random.randint(7, 11)
    for index in range(1, forward_steps + 1):
        progress = index / forward_steps
        eased = 1 - (1 - progress) * (1 - progress)
        next_x = start_x + (overshoot_x - start_x) * eased + random.uniform(-1.2, 1.2)
        if index == forward_steps:
            next_x = overshoot_x
        if next_x < current_x:
            next_x = current_x + random.uniform(0.8, 2.0)
        current_x = next_x
        page.mouse.move(current_x, start_y + random.uniform(-1.8, 1.8), steps=random.randint(3, 7))
        page.wait_for_timeout(random.randint(25, 65))

    pullback_steps = random.randint(3, 5)
    for index in range(1, pullback_steps + 1):
        progress = index / pullback_steps
        next_x = overshoot_x - overshoot * progress + random.uniform(-0.8, 0.8)
        if index == pullback_steps:
            next_x = target_x
        current_x = next_x
        page.mouse.move(current_x, start_y + random.uniform(-1.0, 1.0), steps=random.randint(2, 5))
        page.wait_for_timeout(random.randint(45, 95))

    try:
        knob_left = page.evaluate(
            """() => {
                const node = document.querySelector('.em_slider_knob');
                return node ? (node.style.left || getComputedStyle(node).left) : null;
            }"""
        )
        logger.info("slider knob before mouseup label=%s left=%s expected_distance=%.1f", label, knob_left, distance)
    except Exception as exc:
        logger.warning("failed to read slider knob before mouseup label=%s error=%s", label, exc)

    page.wait_for_timeout(1500)
    page.mouse.up()
    page.wait_for_timeout(1200)
    try:
        knob_left = page.evaluate(
            """() => {
                const node = document.querySelector('.em_slider_knob');
                return node ? (node.style.left || getComputedStyle(node).left) : null;
            }"""
        )
        logger.info("slider knob after mouseup label=%s left=%s expected_distance=%.1f", label, knob_left, distance)
    except Exception as exc:
        logger.warning("failed to read slider knob after mouseup label=%s error=%s", label, exc)
    _click_confirm_if_present(page)
    page.wait_for_timeout(1800)
    return True


def _save_captcha_debug_summary(
    page,
    label: str,
    attempt: int,
    captcha_solver: str,
    debug_dir: Path | None,
) -> None:
    if debug_dir is None:
        return
    prefix = _captcha_debug_prefix(label, attempt, f"summary_{captcha_solver}")
    debug_dir.mkdir(parents=True, exist_ok=True)
    try:
        page.screenshot(path=str(debug_dir / f"{prefix}_page.png"), full_page=True)
    except Exception as exc:
        logger.warning("failed to save captcha debug page screenshot label=%s error=%s", label, exc)
    try:
        data = page.evaluate(
            """() => {
                const selectors = ['.em_box', '.em_widget', '.em_bg', '.em_fullbg', '.em_slice', '.em_slider_knob'];
                const result = {};
                for (const selector of selectors) {
                    const node = document.querySelector(selector);
                    if (!node) {
                        result[selector] = null;
                        continue;
                    }
                    const rect = node.getBoundingClientRect();
                    const style = getComputedStyle(node);
                    result[selector] = {
                        left: rect.left,
                        top: rect.top,
                        width: rect.width,
                        height: rect.height,
                        display: style.display,
                        visibility: style.visibility,
                        opacity: style.opacity,
                        backgroundImage: style.backgroundImage,
                        styleLeft: node.style.left || null
                    };
                }
                return result;
            }"""
        )
    except Exception as exc:
        data = {"error": str(exc)}
    _write_debug_json(
        debug_dir,
        prefix,
        {
            "label": label,
            "attempt": attempt,
            "captcha_solver": captcha_solver,
            "dom": data,
        },
    )


def _eastmoney_gap_distance(
    page,
    label: str,
    distance_scale: float,
    distance_offset: float | None,
    attempt: int,
    debug_dir: Path | None,
    active_capture: dict | None = None,
) -> float | None:
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        logger.warning("opencv is unavailable; cannot solve Eastmoney captcha label=%s error=%s", label, exc)
        return None

    debug_prefix = _captcha_debug_prefix(label, attempt, "cv")
    bg_raw = active_capture.get("bg_raw") if active_capture else None
    full_raw = active_capture.get("full_raw") if active_capture else None
    display_width = active_capture.get("display_width") if active_capture else None
    bg_box = active_capture.get("bg_box") if active_capture else None
    piece_box = active_capture.get("piece_box") if active_capture else None
    bg = _decode_png_bytes(bg_raw) if bg_raw else None
    full = _decode_png_bytes(full_raw) if full_raw else None
    if bg is None or full is None:
        bg, full, display_width = _eastmoney_image_pair(page)
    if bg is None or full is None:
        try:
            page.evaluate(
                """() => {
                    document.querySelectorAll('.em_slice').forEach(node => {
                        node.dataset.codexOldVisibility = node.style.visibility || '';
                        node.style.visibility = 'hidden';
                    });
                }"""
            )
            bg_raw = _element_screenshot(page, ".em_bg.em_show, .em_bg")
            full_raw = _element_screenshot(page, ".em_fullbg.em_show, .em_fullbg")
        finally:
            try:
                page.evaluate(
                    """() => {
                        document.querySelectorAll('.em_slice').forEach(node => {
                            node.style.visibility = node.dataset.codexOldVisibility || '';
                            delete node.dataset.codexOldVisibility;
                        });
                    }"""
                )
            except Exception:
                pass

        if not bg_raw or not full_raw:
            logger.warning("Eastmoney captcha screenshots missing label=%s bg=%s full=%s", label, bool(bg_raw), bool(full_raw))
            return None

        bg = _decode_png_bytes(bg_raw)
        full = _decode_png_bytes(full_raw)
        display_width = None
        if bg is None or full is None:
            return None
    if debug_dir is not None:
        if bg_raw:
            _write_debug_bytes(debug_dir, debug_prefix, "bg", bg_raw)
        if full_raw:
            _write_debug_bytes(debug_dir, debug_prefix, "full", full_raw)
        _write_debug_cv_image(debug_dir, debug_prefix, "bg_cv", bg)
        _write_debug_cv_image(debug_dir, debug_prefix, "full_cv", full)
        page.screenshot(path=str(debug_dir / f"{debug_prefix}_page.png"), full_page=True)
    if bg.shape != full.shape:
        full = cv2.resize(full, (bg.shape[1], bg.shape[0]))

    diff = cv2.absdiff(full, bg)
    _, threshold = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)
    _write_debug_cv_image(debug_dir, debug_prefix, "diff", diff)
    _write_debug_cv_image(debug_dir, debug_prefix, "threshold", threshold)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    expected_y = None
    if bg_box and piece_box and bg_box.get("height"):
        y_scale = bg.shape[0] / float(bg_box["height"])
        expected_y = (float(piece_box["top"]) - float(bg_box["top"])) * y_scale

    candidates: list[tuple[float, int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if expected_y is not None and abs(y - expected_y) > 45:
            continue
        if 15 <= w <= 90 and 15 <= h <= 90 and area >= 80:
            candidates.append((area, x, y, w, h))
    if not candidates:
        logger.warning("Eastmoney captcha gap not found label=%s contour_count=%s", label, len(contours))
        return None

    _, x, y, w, h = max(candidates, key=lambda item: item[0])
    _draw_debug_target(debug_dir, debug_prefix, "target_on_bg", bg, x, (x, y, w, h))
    _draw_debug_target(debug_dir, debug_prefix, "target_on_diff", diff, x, (x, y, w, h))
    _draw_debug_target(debug_dir, debug_prefix, "target_on_threshold", threshold, x, (x, y, w, h))
    scale = float(display_width) / bg.shape[1] if display_width else 1.0
    raw_distance = float(x) * scale
    distance = _normalize_captcha_distance(
        raw_distance,
        distance_scale,
        distance_offset,
        attempt,
        label,
        "cv",
        debug_dir,
        debug_prefix,
        {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "image_scale": scale,
            "raw_distance": raw_distance,
            "contour_count": len(contours),
            "candidate_count": len(candidates),
            "expected_y": expected_y,
            "candidates": [
                {"area": area, "x": cx, "y": cy, "w": cw, "h": ch}
                for area, cx, cy, cw, ch in sorted(candidates, reverse=True)[:8]
            ],
        },
    )
    if distance is None:
        return None
    logger.info(
        "Eastmoney captcha gap label=%s x=%s y=%s w=%s h=%s scale=%.4f raw_distance=%.1f distance=%.1f",
        label,
        x,
        y,
        w,
        h,
        scale,
        raw_distance,
        distance,
    )
    return distance


def _recognizer_gap_distance(
    page,
    label: str,
    distance_scale: float,
    distance_offset: float | None,
    attempt: int,
    debug_dir: Path | None,
) -> float | None:
    try:
        from captcha_recognizer.slider import Slider
    except Exception as exc:
        logger.warning("captcha_recognizer is unavailable label=%s error=%s", label, exc)
        return None

    debug_prefix = _captcha_debug_prefix(label, attempt, "recognizer")
    raw = _element_screenshot(page, ".em_box")
    box = None
    if raw:
        box = page.query_selector(".em_box").bounding_box()
    if not raw:
        raw = _element_screenshot(page, ".em_widget")
        node = page.query_selector(".em_widget")
        box = node.bounding_box() if node else None
    if not raw:
        logger.warning("captcha_recognizer screenshot missing label=%s", label)
        return None
    _write_debug_bytes(debug_dir, debug_prefix, "widget", raw)

    try:
        import cv2

        image = _decode_png_bytes(raw)
        image_width = image.shape[1] if image is not None else None
        result = Slider().identify(source=raw, show=False)
        logger.info("captcha_recognizer raw result label=%s result=%s", label, result)
        if isinstance(result, dict):
            target = result.get("target") or result.get("box") or result.get("position") or result.get("result")
        else:
            target = result
            if isinstance(result, (list, tuple)) and result and isinstance(result[0], (list, tuple, dict)):
                target = result[0]
        if isinstance(target, dict):
            x1 = target.get("x1") or target.get("x") or target.get("left")
        else:
            x1 = target[0] if target and len(target) >= 1 else None
        if x1 is None:
            return None
        scale = float(box["width"]) / image_width if box and image_width else 1.0
        raw_distance = float(x1) * scale
        distance = _normalize_captcha_distance(
            raw_distance,
            distance_scale,
            distance_offset,
            attempt,
            label,
            "recognizer",
            debug_dir,
            debug_prefix,
            {"x": x1, "image_width": image_width, "image_scale": scale, "raw_result": str(result)},
        )
        if distance is None:
            return None
        offset = _attempt_distance_offset(distance_offset, attempt)
        logger.info(
            "captcha_recognizer gap label=%s x=%s scale=%.4f raw_distance=%.1f distance_scale=%.3f distance_offset=%s attempt=%s distance=%.1f",
            label,
            x1,
            scale,
            raw_distance,
            distance_scale,
            offset,
            attempt,
            distance,
        )
        return distance
    except Exception as exc:
        logger.warning("captcha_recognizer failed label=%s error=%s", label, exc)
        return None


def _puzzle_gap_distance(
    page,
    label: str,
    distance_scale: float,
    distance_offset: float | None,
    attempt: int,
    debug_dir: Path | None,
    active_capture: dict | None = None,
) -> float | None:
    try:
        from puzzle_slider_captcha import PuzzleCaptchaSolver
    except Exception as exc:
        logger.warning("PuzzleCaptchaSolver is unavailable label=%s error=%s", label, exc)
        return None

    debug_prefix = _captcha_debug_prefix(label, attempt, "puzzle")
    bg_raw = active_capture.get("bg_raw") if active_capture else None
    piece_raw = active_capture.get("piece_raw") if active_capture else None
    if not bg_raw or not piece_raw:
        bg_raw, piece_raw = _eastmoney_piece_pair(page, label)
    if not bg_raw or not piece_raw:
        return None
    _write_debug_bytes(debug_dir, debug_prefix, "bg", bg_raw)
    _write_debug_bytes(debug_dir, debug_prefix, "piece", piece_raw)

    try:
        solver = PuzzleCaptchaSolver()
        result = solver.handle_bytes(bg_raw, piece_raw)
        x = getattr(result, "x", None)
        y = getattr(result, "y", None)
        confidence = getattr(result, "confidence", None) or getattr(result, "score", None)
        if x is None and isinstance(result, dict):
            x = result.get("x") or result.get("target_x") or result.get("left")
            y = result.get("y") or result.get("target_y") or result.get("top")
            confidence = result.get("confidence") or result.get("score")
        if x is None and isinstance(result, (list, tuple)) and result:
            x = result[0]
            y = result[1] if len(result) > 1 else y
        if x is None:
            logger.warning("PuzzleCaptchaSolver returned no x label=%s result=%s", label, result)
            return None

        raw_distance = float(x)
        distance = _normalize_captcha_distance(
            raw_distance,
            distance_scale,
            distance_offset,
            attempt,
            label,
            "puzzle",
            debug_dir,
            debug_prefix,
            {"x": x, "y": y, "confidence": confidence, "raw_result": str(result)},
        )
        if distance is None:
            return None
        offset = _attempt_distance_offset(distance_offset, attempt)
        logger.info(
            "PuzzleCaptchaSolver gap label=%s x=%s y=%s confidence=%s raw_distance=%.1f distance_scale=%.3f distance_offset=%s attempt=%s distance=%.1f",
            label,
            x,
            y,
            confidence,
            raw_distance,
            distance_scale,
            offset,
            attempt,
            distance,
        )
        return distance
    except Exception as exc:
        logger.warning("PuzzleCaptchaSolver failed label=%s error=%s", label, exc)
        return None


def _try_eastmoney_slider(
    page,
    label: str,
    captcha_solver: str,
    distance_scale: float,
    distance_offset: float | None,
    attempt: int,
    debug_dir: Path | None,
) -> bool:
    _, handle = _find_first_element(page, [".em_slider_knob", ".em_slider .em_slider_knob"])
    if not handle:
        return False
    logger.info("Eastmoney slider widget found label=%s", label)
    solver_order = [captcha_solver]
    if captcha_solver == "auto":
        solver_order = ["puzzle", "cv", "recognizer"]
    elif captcha_solver == "none":
        solver_order = []

    for solver in solver_order:
        active_capture = _capture_active_slider_state(page, handle, label, attempt, solver, debug_dir)
        page.wait_for_timeout(700)
        if solver == "puzzle":
            distance = _puzzle_gap_distance(page, label, distance_scale, distance_offset, attempt, debug_dir, active_capture)
        elif solver == "recognizer":
            distance = _recognizer_gap_distance(page, label, distance_scale, distance_offset, attempt, debug_dir)
        elif solver == "cv":
            distance = _eastmoney_gap_distance(page, label, distance_scale, distance_offset, attempt, debug_dir, active_capture)
        else:
            distance = None
        if distance is None:
            logger.info("captcha solver did not produce usable distance label=%s solver=%s", label, solver)
            continue
        logger.info("captcha solver selected label=%s solver=%s distance=%.1f", label, solver, distance)
        return _drag_handle(page, handle, distance, label)
    return False


def _try_auto_slider(
    page,
    label: str,
    captcha_solver: str,
    distance_scale: float,
    distance_offset: float | None,
    attempt: int,
    debug_dir: Path | None,
) -> bool:
    if _try_eastmoney_slider(page, label, captcha_solver, distance_scale, distance_offset, attempt, debug_dir):
        return True
    if captcha_solver not in {"cv", "auto"}:
        return False

    bg_selectors = [".bg_img", "canvas.bg_img", "canvas[class*='bg']", ".geetest_canvas_bg", ".tc-bg-img canvas"]
    piece_selectors = [
        ".bg_slider",
        "canvas.bg_slider",
        "canvas[class*='slider']",
        ".geetest_canvas_slice",
        ".tc-slider-normal",
    ]
    handle_selectors = [
        ".sc_net_slider_icon",
        ".em_slider_knob",
        "[class*='slider_icon']",
        "[class*='slider-btn']",
        "[class*='sliderBtn']",
        ".geetest_slider_button",
        ".tc-slider-normal",
    ]

    try:
        import cv2

        cv2  # keep optional import local
    except Exception as exc:
        logger.warning("opencv is unavailable; cannot auto slide captcha label=%s error=%s", label, exc)
        return False

    bg_selector = next((selector for selector in bg_selectors if _canvas_data_url(page, selector)), None)
    piece_selector = next((selector for selector in piece_selectors if _canvas_data_url(page, selector)), None)
    handle_selector, handle = _find_first_element(page, handle_selectors)
    if bg_selector and not piece_selector and handle:
        _reveal_slider_piece(page, handle, label)
        bg_selector = next((selector for selector in bg_selectors if _canvas_data_url(page, selector)), None)
        piece_selector = next((selector for selector in piece_selectors if _canvas_data_url(page, selector)), None)
        handle_selector, handle = _find_first_element(page, handle_selectors)
    if not bg_selector or not piece_selector or not handle:
        logger.warning(
            "slider elements not found label=%s bg=%s piece=%s handle=%s",
            label,
            bg_selector,
            piece_selector,
            handle_selector,
        )
        return False

    try:
        import cv2

        bg_png = _canvas_data_url(page, bg_selector)
        piece_png = _canvas_data_url(page, piece_selector)
        if not bg_png or not piece_png:
            return False
        bg = cv2.Canny(_decode_png(bg_png), 100, 200)
        piece = cv2.Canny(_decode_png(piece_png), 100, 200)
        result = cv2.matchTemplate(bg, piece, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, max_loc = cv2.minMaxLoc(result)
        x_offset = max_loc[0] + 14

        bg_box = page.query_selector(bg_selector).bounding_box()
        handle_box = handle.bounding_box()
        if not bg_box or not handle_box:
            return False
        distance = x_offset * bg_box["width"] / max(1, bg.shape[1])
        distance = max(20, distance)
        logger.info("auto slider label=%s distance=%.1f confidence=%.4f", label, distance, confidence)

        start_x = handle_box["x"] + handle_box["width"] / 2
        start_y = handle_box["y"] + handle_box["height"] / 2
        page.mouse.move(start_x, start_y)
        page.mouse.down()
        current_x = start_x
        for move in _slider_tracks(distance):
            current_x += move
            page.mouse.move(current_x, start_y + random.uniform(-1.5, 1.5), steps=1)
            page.wait_for_timeout(random.randint(8, 22))
        page.mouse.up()
        page.wait_for_timeout(1200)
        _click_confirm_if_present(page)
        page.wait_for_timeout(1800)
        return True
    except Exception as exc:
        logger.warning("auto slider failed label=%s error=%s", label, exc)
        return False


def _resolve_risk_page(
    page,
    html: str,
    label: str,
    headed: bool,
    max_retries: int,
    retry_wait: float,
    auto_captcha: bool,
    manual_fallback: bool,
    captcha_solver: str,
    captcha_distance_scale: float,
    captcha_distance_offset: float | None,
    captcha_debug_dir: Path | None,
) -> str:
    for attempt in range(1, max_retries + 1):
        if not _is_risk_page(html):
            return html
        logger.warning("risk/captcha page detected label=%s attempt=%s/%s", label, attempt, max_retries)
        _save_captcha_debug_summary(page, label, attempt, captcha_solver, captcha_debug_dir)
        solved_by_auto = auto_captcha and _try_auto_slider(
            page,
            label,
            captcha_solver,
            captcha_distance_scale,
            captcha_distance_offset,
            attempt,
            captcha_debug_dir,
        )
        page.wait_for_timeout(2500)
        html = page.content()
        if not _is_risk_page(html):
            return html
        if solved_by_auto:
            logger.warning("auto captcha attempt did not pass label=%s attempt=%s/%s", label, attempt, max_retries)
        if attempt >= max_retries:
            break
        if retry_wait > 0:
            wait_seconds = retry_wait * min(3.0, 1.0 + (attempt - 1) * 0.35) + random.uniform(0.5, 2.5)
            logger.info("wait %.1fs before next captcha retry label=%s", wait_seconds, label)
            time.sleep(wait_seconds)
        page.reload(wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(1500)
        html = page.content()
    if _is_risk_page(html):
        if headed and manual_fallback:
            input(f"[{label}] Auto captcha failed after {max_retries} attempts. Finish it in browser, then press Enter here...")
            page.wait_for_timeout(1500)
            html = page.content()
            if not _is_risk_page(html):
                return html
        raise RuntimeError(f"Risk/captcha page still active after {max_retries} attempts: {label}")
    return html


def _load_page_with_retry(
    page,
    url: str,
    label: str,
    headed: bool,
    max_retries: int,
    retry_wait: float,
    auto_captcha: bool,
    manual_fallback: bool,
    captcha_solver: str,
    captcha_distance_scale: float,
    captcha_distance_offset: float | None,
    captcha_debug_dir: Path | None,
    settle_ms: int = 1500,
    recover_url: str | None = None,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(settle_ms)
            html = page.content()
            if _is_forbidden_page(html):
                logger.warning(
                    "403 forbidden detected label=%s url=%s attempt=%s/%s recover_url=%s",
                    label,
                    url,
                    attempt,
                    max_retries,
                    recover_url,
                )
                if recover_url:
                    try:
                        page.goto(recover_url, wait_until="domcontentloaded", timeout=60000)
                        page.wait_for_timeout(max(2000, settle_ms))
                    except Exception as recover_exc:
                        logger.warning("403 recover navigation failed label=%s recover_url=%s error=%s", label, recover_url, recover_exc)
                if attempt < max_retries and retry_wait > 0:
                    time.sleep(retry_wait)
                continue
            return _resolve_risk_page(
                page,
                html,
                label,
                headed,
                max_retries,
                retry_wait,
                auto_captcha,
                manual_fallback,
                captcha_solver,
                captcha_distance_scale,
                captcha_distance_offset,
                captcha_debug_dir,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "load failed label=%s url=%s attempt=%s/%s error=%s",
                label,
                url,
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries and retry_wait > 0:
                time.sleep(retry_wait)
    raise RuntimeError(f"Load failed after {max_retries} attempts: {url}") from last_error


def _parse_list_html(html: str, base_url: str, bar_name: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    bar_code = bar_code_from_url(base_url)
    rows: list[dict] = []
    for a in soup.select("tr.listitem td .title a[href], .listitem .title a[href], a[href*='/news,']"):
        href = a.get("href") or ""
        title = a.get_text(" ", strip=True)
        if not href or not title:
            continue
        if href.startswith("//"):
            full_url = "https:" + href
        elif href.startswith("/"):
            full_url = "https://guba.eastmoney.com" + href
        else:
            full_url = href
        if "guba.eastmoney.com/news," not in full_url:
            continue
        if bar_code and f"/news,{bar_code}," not in full_url:
            continue
        parent = a.find_parent("tr")
        read_count = _safe_int(parent.select_one(".read").get_text(strip=True)) if parent and parent.select_one(".read") else None
        comment_count = _safe_int(parent.select_one(".reply").get_text(strip=True)) if parent and parent.select_one(".reply") else None
        rows.append(
            {
                "post_id": _post_id(full_url),
                "source": "eastmoney_guba",
                "bar_name": bar_name,
                "publish_time": None,
                "trade_date": datetime.now().date(),
                "title": title[:500],
                "content": title,
                "read_count": read_count,
                "comment_count": comment_count,
                "like_count": None,
                "author": None,
                "url": full_url,
            }
        )
    dedup = {row["post_id"]: row for row in rows}
    return list(dedup.values())


def _parse_detail_html(html: str) -> dict:
    publish_time = pd.to_datetime(_json_field(html, "post_publish_time"), errors="coerce")
    title = _json_field(html, "post_title")
    content = _json_field(html, "post_content") or _json_field(html, "post_abstract")
    if not content:
        soup = BeautifulSoup(html, "html.parser")
        node = soup.select_one(".newstext, .stockcodec, .article-body")
        content = node.get_text(" ", strip=True) if node else ""
    data = {
        "title": title,
        "content": content,
        "read_count": _json_int(html, "post_click_count"),
        "comment_count": _json_int(html, "post_comment_count"),
        "like_count": _json_int(html, "post_like_count"),
    }
    if publish_time is not None and not pd.isna(publish_time):
        data["publish_time"] = publish_time.to_pydatetime()
        data["trade_date"] = publish_time.date()
    return {k: v for k, v in data.items() if v not in (None, "")}


def _comment_id(post_id: str, content: str, author: str | None, publish_time: object | None, index: int) -> str:
    key = f"{post_id}|{author or ''}|{publish_time or ''}|{content}|{index}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _parse_comment_time(text: str) -> datetime | None:
    if not text:
        return None
    match = re.search(r"(20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)", text)
    if match:
        value = match.group(1)
        return pd.to_datetime(value, errors="coerce").to_pydatetime()
    match = re.search(r"(\d{2}-\d{2}\s+\d{2}:\d{2})", text)
    if match:
        value = f"{datetime.now().year}-{match.group(1)}"
        return pd.to_datetime(value, errors="coerce").to_pydatetime()
    return None


def _parse_comments_from_html(html: str, post_id: str, bar_name: str, url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    candidates = soup.select(
        ".reply_item, .comment_item, .replyItem, .commentItem, "
        ".reply-list li, .comment-list li, [class*='reply_item'], [class*='comment_item']"
    )
    rows: list[dict] = []
    seen: set[str] = set()
    for index, node in enumerate(candidates):
        text = node.get_text(" ", strip=True)
        if len(text) < 2:
            continue
        content_node = node.select_one(".reply_content, .comment_content, [class*='content'], [class*='text']")
        content = content_node.get_text(" ", strip=True) if content_node else text
        if len(content) < 2 or content in seen:
            continue
        seen.add(content)
        author_node = node.select_one(".author, .user, .name, a[href*='i.eastmoney.com']")
        author = author_node.get_text(" ", strip=True)[:100] if author_node else None
        publish_time = _parse_comment_time(text)
        like_match = re.search(r"(?:like|\u8d5e)[^\d]*(\d+)", text, re.I)
        reply_match = re.search(r"(?:reply|\u56de\u590d)[^\d]*(\d+)", text, re.I)
        rows.append(
            {
                "comment_id": _comment_id(post_id, content, author, publish_time, index),
                "post_id": post_id,
                "source": "eastmoney_guba",
                "bar_name": bar_name,
                "publish_time": publish_time,
                "trade_date": publish_time.date() if publish_time else None,
                "author": author,
                "content": content,
                "like_count": _safe_int(like_match.group(1)) if like_match else None,
                "reply_count": _safe_int(reply_match.group(1)) if reply_match else None,
                "url": url,
            }
        )
    return rows


def collect_guba_history_playwright(
    start_page: int,
    end_page: int,
    bar_name: str | None = None,
    sleep_seconds: float = 1.0,
    detail_sleep_seconds: float = 0.2,
    batch_pages: int = 5,
    user_data_dir: str = "data/browser/eastmoney",
    headed: bool = True,
    pause_first_page: bool = True,
    fetch_detail: bool = True,
    require_publish_time: bool = True,
    fetch_comments: bool = False,
    min_comments: int = 3,
    max_comment_pages: int = 2,
    max_retries: int = 8,
    retry_wait: float = 30.0,
    auto_captcha: bool = True,
    manual_captcha_fallback: bool = True,
    captcha_solver: str = "auto",
    captcha_distance_scale: float = 1.0,
    captcha_distance_offset: float | None = None,
    captcha_debug_dir: str | None = None,
) -> int:
    from playwright.sync_api import sync_playwright

    bars = _selected_bars(bar_name)
    if not bars:
        raise ValueError(f"No guba bar matched: {bar_name}")

    init_database()
    total = 0
    comment_total = 0
    user_data_path = project_path(user_data_dir)
    user_data_path.mkdir(parents=True, exist_ok=True)
    captcha_debug_path = project_path(captcha_debug_dir) if captcha_debug_dir else None
    if captcha_debug_path:
        captcha_debug_path.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_path),
            headless=not headed,
            viewport={"width": 1360, "height": 900},
            locale="zh-CN",
        )
        list_page = context.new_page()
        detail_page = context.new_page()

        try:
            for bar in bars:
                buffer: list[dict] = []
                empty_pages = 0
                logger.info("start playwright guba history bar=%s pages=%s-%s", bar["bar_name"], start_page, end_page)
                for page_no in range(start_page, end_page + 1):
                    comment_buffer: list[dict] = []
                    url = page_url(bar["url"], page_no)
                    try:
                        html = _load_page_with_retry(
                            list_page,
                            url,
                            label=f"{bar['bar_name']} page {page_no}",
                            headed=headed,
                            max_retries=max_retries,
                            retry_wait=retry_wait,
                            auto_captcha=auto_captcha,
                            manual_fallback=manual_captcha_fallback,
                            captcha_solver=captcha_solver,
                            captcha_distance_scale=captcha_distance_scale,
                            captcha_distance_offset=captcha_distance_offset,
                            captcha_debug_dir=captcha_debug_path,
                            settle_ms=1500,
                            recover_url=bar["url"],
                        )
                        if pause_first_page and page_no == start_page and headed:
                            input("Browser is open. If it shows captcha, wait for auto handling or finish it, then press Enter...")
                            pause_first_page = False

                        rows = _parse_list_html(html, bar["url"], bar["bar_name"])
                        if not rows:
                            empty_pages += 1
                            logger.warning(
                                "empty playwright guba page bar=%s page=%s empty_pages=%s",
                                bar["bar_name"],
                                page_no,
                                empty_pages,
                            )
                        else:
                            empty_pages = 0
                            logger.info("playwright guba page bar=%s page=%s list_rows=%s", bar["bar_name"], page_no, len(rows))

                        if fetch_detail:
                            detailed_rows: list[dict] = []
                            for row in rows:
                                try:
                                    detail_html = _load_page_with_retry(
                                        detail_page,
                                        row["url"],
                                        label=f"detail {row['post_id']}",
                                        headed=headed,
                                        max_retries=max_retries,
                                        retry_wait=retry_wait,
                                        auto_captcha=auto_captcha,
                                        manual_fallback=manual_captcha_fallback,
                                        captcha_solver=captcha_solver,
                                        captcha_distance_scale=captcha_distance_scale,
                                        captcha_distance_offset=captcha_distance_offset,
                                        captcha_debug_dir=captcha_debug_path,
                                        settle_ms=800,
                                        recover_url=url,
                                    )
                                    row.update(_parse_detail_html(detail_html))
                                    if fetch_comments and (row.get("comment_count") or 0) >= min_comments:
                                        comment_rows = _parse_comments_from_html(
                                            detail_html,
                                            row["post_id"],
                                            row["bar_name"],
                                            row["url"],
                                        )
                                        if comment_rows:
                                            logger.info("parsed comments post_id=%s rows=%s", row["post_id"], len(comment_rows))
                                            comment_buffer.extend(comment_rows)
                                except Exception as exc:
                                    logger.warning("failed detail url=%s error=%s", row["url"], exc)
                                if detail_sleep_seconds > 0:
                                    time.sleep(detail_sleep_seconds)
                                if require_publish_time and not row.get("publish_time"):
                                    logger.warning("skip row without publish_time url=%s title=%s", row["url"], row.get("title"))
                                    continue
                                detailed_rows.append(row)
                            rows = detailed_rows

                        buffer.extend(rows)
                        if buffer and (page_no - start_page + 1) % batch_pages == 0:
                            total += upsert_dataframe(pd.DataFrame(buffer), "sentiment_guba_raw", ["post_id"])
                            buffer = []
                        if comment_buffer:
                            comment_total += upsert_dataframe(pd.DataFrame(comment_buffer), "sentiment_guba_comment_raw", ["comment_id"])
                        if empty_pages >= 5:
                            logger.warning("stop bar=%s after 5 continuous empty pages at page=%s", bar["bar_name"], page_no)
                            break
                    except Exception as exc:
                        logger.exception("failed playwright guba bar=%s page=%s: %s", bar["bar_name"], page_no, exc)
                    finally:
                        if sleep_seconds > 0:
                            time.sleep(sleep_seconds)

                if buffer:
                    total += upsert_dataframe(pd.DataFrame(buffer), "sentiment_guba_raw", ["post_id"])
                logger.info(
                    "finish playwright guba history bar=%s post_total_so_far=%s comment_total_so_far=%s",
                    bar["bar_name"],
                    total,
                    comment_total,
                )
        finally:
            context.close()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Eastmoney Guba history with Playwright browser.")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=50)
    parser.add_argument("--bar-name")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--detail-sleep", type=float, default=0.2)
    parser.add_argument("--batch-pages", type=int, default=5)
    parser.add_argument("--user-data-dir", default="data/browser/eastmoney")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-pause-first-page", action="store_true")
    parser.add_argument("--no-detail", action="store_true")
    parser.add_argument("--allow-missing-time", action="store_true", help="Keep rows even if detail publish_time is missing.")
    parser.add_argument("--fetch-comments", action="store_true", help="Fetch comments for posts with enough comments.")
    parser.add_argument("--min-comments", type=int, default=3, help="Only fetch comments when post comment_count >= this value.")
    parser.add_argument("--max-comment-pages", type=int, default=2, help="Reserved for future comment pagination support.")
    parser.add_argument("--max-retries", type=int, default=8, help="Retry list/detail pages after captcha, network, or browser failures.")
    parser.add_argument("--retry-wait", type=float, default=30.0, help="Seconds to wait between retries or captcha rechecks.")
    parser.add_argument("--no-auto-captcha", action="store_true", help="Disable best-effort automatic slider captcha handling.")
    parser.add_argument("--no-manual-captcha-fallback", action="store_true", help="Do not pause for manual captcha if auto handling fails.")
    parser.add_argument(
        "--captcha-solver",
        choices=["auto", "puzzle", "recognizer", "cv", "none"],
        default="auto",
        help="Captcha solver to use. Auto tries PuzzleCaptchaSolver, OpenCV, then captcha_recognizer.",
    )
    parser.add_argument("--captcha-distance-scale", type=float, default=1.0, help="Multiply captcha drag distance by this factor.")
    parser.add_argument(
        "--captcha-distance-offset",
        type=float,
        default=None,
        help="Add this many pixels to captcha drag distance. Default auto-tries several offsets.",
    )
    parser.add_argument(
        "--captcha-debug-dir",
        default=None,
        help="Directory to save captcha screenshots and distance metadata for debugging.",
    )
    args = parser.parse_args()
    print(
        collect_guba_history_playwright(
            start_page=args.start_page,
            end_page=args.end_page,
            bar_name=args.bar_name,
            sleep_seconds=args.sleep,
            detail_sleep_seconds=args.detail_sleep,
            batch_pages=args.batch_pages,
            user_data_dir=args.user_data_dir,
            headed=not args.headless,
            pause_first_page=not args.no_pause_first_page,
            fetch_detail=not args.no_detail,
            require_publish_time=not args.allow_missing_time,
            fetch_comments=args.fetch_comments,
            min_comments=args.min_comments,
            max_comment_pages=args.max_comment_pages,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
            auto_captcha=not args.no_auto_captcha,
            manual_captcha_fallback=not args.no_manual_captcha_fallback,
            captcha_solver=args.captcha_solver,
            captcha_distance_scale=args.captcha_distance_scale,
            captcha_distance_offset=args.captcha_distance_offset,
            captcha_debug_dir=args.captcha_debug_dir,
        )
    )


if __name__ == "__main__":
    main()
