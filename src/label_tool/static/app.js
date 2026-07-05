const chart = echarts.init(document.getElementById("chart"));
const state = {
  freq: "daily",
  kline: [],
  labels: [],
  candidates: [],
  meta: null,
  editingId: null,
  selectedRange: null,
  lineMode: false,
  lineStartIndex: null,
  dragStartIndex: null,
  dragPreviewIndex: null,
  dataZoom: { start: 65, end: 100 },
  autoZoomPending: true,
};

const reasons = [
  "panic_selloff", "panic_liquidation", "liquidity_crisis", "policy_support",
  "policy_expectation", "policy_tightening", "reopening_trade", "trade_war",
  "covid", "global_risk", "fed_hike_expect", "valuation_low", "valuation_high",
  "growth_overvalued", "high_valuation", "uncertain", "weekly_boll_lower",
  "monthly_boll_lower", "daily_boll_lower", "oversold_rebound", "kdj_oversold",
  "weekly_kdj_golden_cross", "macd_bottom_divergence", "volume_shrink",
  "volume_expand", "down_with_volume", "lower_shadow", "technical_pullback",
  "weekly_boll_upper", "monthly_boll_upper", "daily_boll_upper", "kdj_overbought",
  "weekly_kdj_dead_cross", "macd_top_divergence", "euphoria",
  "rebound_exhaustion", "upper_shadow", "new_high_area", "intraday_v_reversal",
  "intraday_volume_spike", "intraday_boll_break", "intraday_kdj_cross",
  "intraday_macd_cross", "intraday_failed_breakout", "intraday_panic_drop",
  "intraday_exhaustion"
];

const zh = {
  region_type: { bottom: "底部", top: "顶部" },
  cycle: { short: "短期", medium: "中期", long: "长期", intraday: "分时" },
  level: { minor: "小级别", swing: "波段级别", major: "大级别" },
  label_freq: { daily: "日K", weekly: "周K", monthly: "月K", intraday: "小时K" },
  usable_for_signal: { "1": "用于训练", "0": "仅观察" },
};

const reasonZh = {
  candidate_bottom: "候选底部",
  candidate_top: "候选顶部",
  panic_selloff: "恐慌抛售",
  panic_liquidation: "流动性踩踏",
  liquidity_crisis: "流动性危机",
  policy_support: "政策支撑",
  policy_expectation: "政策预期",
  policy_tightening: "政策收紧",
  reopening_trade: "重启交易",
  trade_war: "贸易摩擦",
  covid: "疫情冲击",
  global_risk: "全球风险",
  fed_hike_expect: "美联储加息预期",
  valuation_low: "估值低位",
  valuation_high: "估值高位",
  growth_overvalued: "成长高估",
  high_valuation: "高估值",
  uncertain: "不确定",
  weekly_boll_lower: "周线BOLL下轨",
  monthly_boll_lower: "月线BOLL下轨",
  daily_boll_lower: "日线BOLL下轨",
  oversold_rebound: "超跌反弹",
  kdj_oversold: "KDJ超卖",
  weekly_kdj_golden_cross: "周线KDJ金叉",
  macd_bottom_divergence: "MACD底背离",
  volume_shrink: "缩量",
  volume_expand: "放量",
  down_with_volume: "放量下跌",
  lower_shadow: "下影线",
  technical_pullback: "技术回踩",
  weekly_boll_upper: "周线BOLL上轨",
  monthly_boll_upper: "月线BOLL上轨",
  daily_boll_upper: "日线BOLL上轨",
  kdj_overbought: "KDJ超买",
  weekly_kdj_dead_cross: "周线KDJ死叉",
  macd_top_divergence: "MACD顶背离",
  euphoria: "情绪亢奋",
  rebound_exhaustion: "反弹衰竭",
  upper_shadow: "上影线",
  new_high_area: "新高区域",
  intraday_v_reversal: "分时V形反转",
  intraday_volume_spike: "分时放量",
  intraday_boll_break: "分时BOLL突破",
  intraday_kdj_cross: "分时KDJ交叉",
  intraday_macd_cross: "分时MACD交叉",
  intraday_failed_breakout: "分时突破失败",
  intraday_panic_drop: "分时恐慌下跌",
  intraday_exhaustion: "分时衰竭",
};

function $(id) {
  return document.getElementById(id);
}

function labelZh(type, value) {
  return zh[type]?.[String(value)] || String(value || "");
}

function reasonsDisplay(value) {
  return String(value || "")
    .split("|")
    .map((x) => x.trim())
    .filter(Boolean)
    .map((x) => reasonZh[x] || x)
    .join(" | ");
}

function setStatus(text) {
  $("status").textContent = text || "";
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || "请求失败");
  return data;
}

function toNum(v) {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function labelFreqFromChart() {
  return state.freq;
}

function defaultCycle(freq) {
  if (freq === "intraday") return "intraday";
  if (freq === "weekly") return "medium";
  if (freq === "monthly") return "long";
  return "short";
}

function defaultLevel(freq) {
  if (freq === "monthly") return "major";
  return "swing";
}

function regionPrefix(type) {
  return type === "top" ? "T" : "B";
}

function yyyymm(date) {
  return String(date || "").slice(0, 7).replace("-", "");
}

function nextRegionId(item) {
  const prefix = regionPrefix(item.region_type);
  const month = yyyymm(item.start_date);
  const base = `${prefix}${month}_${item.label_freq}_${item.cycle}`;
  let max = 0;
  for (const label of state.labels) {
    const id = label.region_id || "";
    if (!id.startsWith(base + "_")) continue;
    const n = Number(id.slice(base.length + 1));
    if (Number.isFinite(n)) max = Math.max(max, n);
  }
  return `${base}_${String(max + 1).padStart(2, "0")}`;
}

function normalizeRange(startIndex, endIndex) {
  const a = Math.max(0, Math.min(startIndex, endIndex));
  const b = Math.min(state.kline.length - 1, Math.max(startIndex, endIndex));
  const start = state.kline[a];
  const end = state.kline[b];
  return {
    start_date: start.period_start || start.trade_date,
    end_date: end.period_end || end.trade_date,
  };
}

function dateValue(value, endOfDay = false) {
  const text = String(value || "").trim().replace(/\//g, "-");
  if (!text) return null;
  const normalized = /^\d{4}-\d{2}-\d{2}$/.test(text)
    ? `${text}T${endOfDay ? "23:59:59" : "00:00:00"}`
    : text.replace(" ", "T");
  const parsed = Date.parse(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

function indexRangeForItem(item) {
  if (!state.kline.length || !item.start_date || !item.end_date) return null;
  const startValue = dateValue(item.start_date);
  const endValue = dateValue(item.end_date, true);
  if (startValue === null || endValue === null) return null;

  let startIndex = -1;
  let endIndex = -1;
  for (let i = 0; i < state.kline.length; i += 1) {
    const row = state.kline[i];
    const rowEnd = dateValue(row.period_end || row.trade_date, true);
    if (rowEnd !== null && rowEnd >= startValue) {
      startIndex = i;
      break;
    }
  }
  for (let i = state.kline.length - 1; i >= 0; i -= 1) {
    const row = state.kline[i];
    const rowStart = dateValue(row.period_start || row.trade_date);
    if (rowStart !== null && rowStart <= endValue) {
      endIndex = i;
      break;
    }
  }
  if (startIndex < 0 || endIndex < 0 || startIndex > endIndex) return null;
  return { startIndex, endIndex };
}

function axisRangeForItem(item) {
  const range = indexRangeForItem(item);
  if (!range) return null;
  return {
    start: state.kline[range.startIndex].trade_date,
    end: state.kline[range.endIndex].trade_date,
  };
}

function setZoomForIndexRange(startIndex, endIndex, padding = 6) {
  if (!state.kline.length) return;
  const last = state.kline.length - 1;
  const start = Math.max(0, Math.min(startIndex, endIndex) - padding);
  const end = Math.min(last, Math.max(startIndex, endIndex) + padding);
  if (last <= 0) {
    state.dataZoom = { start: 0, end: 100 };
    return;
  }
  state.dataZoom = {
    start: Math.max(0, Math.min(100, (start / last) * 100)),
    end: Math.max(0, Math.min(100, (end / last) * 100)),
  };
}

function zoomToItem(item) {
  const range = indexRangeForItem(item);
  if (!range) return;
  setZoomForIndexRange(range.startIndex, range.endIndex);
  renderChart(false);
}

function applyInitialLabelZoom() {
  if (!state.autoZoomPending || $("startDate").value || $("endDate").value) return;
  let startIndex = Infinity;
  let endIndex = -Infinity;
  for (const label of state.labels) {
    if (label.label_freq && label.label_freq !== state.freq) continue;
    const range = indexRangeForItem(label);
    if (!range) continue;
    startIndex = Math.min(startIndex, range.startIndex);
    endIndex = Math.max(endIndex, range.endIndex);
  }
  if (!Number.isFinite(startIndex) || !Number.isFinite(endIndex)) return;
  setZoomForIndexRange(startIndex, endIndex);
  state.autoZoomPending = false;
}

function colorFor(label) {
  if (String(label.usable_for_signal) === "0") return "#9ca3af";
  const key = `${label.cycle}_${label.region_type}`;
  return {
    short_bottom: "#60a5fa",
    medium_bottom: "#2563eb",
    long_bottom: "#1e3a8a",
    intraday_bottom: "#38bdf8",
    short_top: "#fdba74",
    medium_top: "#f97316",
    long_top: "#dc2626",
    intraday_top: "#f59e0b",
  }[key] || "#6b7280";
}

function buildMarkAreas() {
  const areas = [];
  for (const label of state.labels) {
    const axisRange = axisRangeForItem(label);
    if (!axisRange) continue;
    const color = colorFor(label);
    areas.push([
      {
        xAxis: axisRange.start,
        itemStyle: { color: `${color}33`, borderColor: color, borderType: String(label.usable_for_signal) === "0" ? "dashed" : "solid" },
        label: {
          show: true,
          formatter: `${label.region_id}\n${label.cycle}/${label.level}/C${label.confidence}`,
          color,
          fontSize: 11,
        },
      },
      { xAxis: axisRange.end },
    ]);
  }
  if ($("showCandidates").checked) {
    for (const candidate of state.candidates) {
      const axisRange = axisRangeForItem(candidate);
      if (!axisRange) continue;
      const color = colorFor(candidate);
      areas.push([
        {
          xAxis: axisRange.start,
          itemStyle: { color: `${color}18`, borderColor: color, borderType: "dashed" },
          label: {
            show: true,
            formatter: `候选\n${candidate.cycle}/${candidate.region_type}`,
            color,
            fontSize: 11,
          },
        },
        { xAxis: axisRange.end },
      ]);
    }
  }
  if (state.lineMode && state.lineStartIndex !== null && state.kline[state.lineStartIndex]) {
    const row = state.kline[state.lineStartIndex];
    areas.push([
      {
        xAxis: row.trade_date,
        itemStyle: { color: "#11182722", borderColor: "#111827", borderType: "dashed" },
        label: { show: true, formatter: "起点", color: "#111827", fontSize: 11 },
      },
      { xAxis: row.trade_date },
    ]);
  }
  if (state.lineMode && state.dragStartIndex !== null && state.dragPreviewIndex !== null) {
    const range = normalizeRange(state.dragStartIndex, state.dragPreviewIndex);
    const axisRange = axisRangeForItem(range);
    if (!axisRange) return areas;
    areas.push([
      {
        xAxis: axisRange.start,
        itemStyle: { color: "#11182722", borderColor: "#111827", borderType: "dashed" },
        label: { show: true, formatter: "画线范围", color: "#111827", fontSize: 11 },
      },
      { xAxis: axisRange.end },
    ]);
  }
  return areas;
}

function buildOption() {
  const dates = state.kline.map((x) => x.trade_date);
  const candles = state.kline.map((x) => [toNum(x.open), toNum(x.close), toNum(x.low), toNum(x.high)]);
  const volume = state.kline.map((x) => toNum(x.volume));
  const mainIndicator = $("mainIndicator").value;
  const subIndicator = $("subIndicator").value;
  const series = [
    {
      name: "K线",
      type: "candlestick",
      data: candles,
      xAxisIndex: 0,
      yAxisIndex: 0,
      itemStyle: { color: "#ef4444", color0: "#16a34a", borderColor: "#ef4444", borderColor0: "#16a34a" },
      markArea: { silent: true, data: buildMarkAreas() },
    },
    { name: "VOL", type: "bar", data: volume, xAxisIndex: 1, yAxisIndex: 1, itemStyle: { color: "#94a3b8" } },
    { name: "VOL MA5", type: "line", data: state.kline.map((x) => toNum(x.vol_ma5)), xAxisIndex: 1, yAxisIndex: 1, symbol: "none", lineStyle: { width: 1 } },
    { name: "VOL MA10", type: "line", data: state.kline.map((x) => toNum(x.vol_ma10)), xAxisIndex: 1, yAxisIndex: 1, symbol: "none", lineStyle: { width: 1 } },
  ];

  if (mainIndicator === "ma") {
    for (const ma of ["ma5", "ma10", "ma20", "ma60", "ma120", "ma250"]) {
      series.push({ name: ma.toUpperCase(), type: "line", data: state.kline.map((x) => toNum(x[ma])), xAxisIndex: 0, yAxisIndex: 0, symbol: "none", lineStyle: { width: 1 } });
    }
  } else if (mainIndicator === "boll") {
    const n = $("bollParam").value === "3" ? "3" : "2";
    const keys = [`boll_upper_${n}`, "boll_mid", `boll_lower_${n}`];
    for (const key of keys) {
      series.push({ name: key, type: "line", data: state.kline.map((x) => toNum(x[key])), xAxisIndex: 0, yAxisIndex: 0, symbol: "none", lineStyle: { width: 1 } });
    }
  }

  if (subIndicator === "macd") {
    series.push({ name: "MACD", type: "bar", data: state.kline.map((x) => toNum(x.macd_bar)), xAxisIndex: 2, yAxisIndex: 2, itemStyle: { color: (p) => p.value >= 0 ? "#ef4444" : "#16a34a" } });
    series.push({ name: "DIF", type: "line", data: state.kline.map((x) => toNum(x.macd_dif)), xAxisIndex: 2, yAxisIndex: 2, symbol: "none", lineStyle: { width: 1 } });
    series.push({ name: "DEA", type: "line", data: state.kline.map((x) => toNum(x.macd_dea)), xAxisIndex: 2, yAxisIndex: 2, symbol: "none", lineStyle: { width: 1 } });
  } else if (subIndicator === "kdj") {
    for (const key of ["kdj_k", "kdj_d", "kdj_j"]) {
      series.push({ name: key.toUpperCase(), type: "line", data: state.kline.map((x) => toNum(x[key])), xAxisIndex: 2, yAxisIndex: 2, symbol: "none", lineStyle: { width: 1 } });
    }
  } else if (subIndicator === "rsi") {
    for (const key of ["rsi6", "rsi12", "rsi24"]) {
      series.push({ name: key.toUpperCase(), type: "line", data: state.kline.map((x) => toNum(x[key])), xAxisIndex: 2, yAxisIndex: 2, symbol: "none", lineStyle: { width: 1 } });
    }
  } else if (subIndicator === "cci") {
    series.push({ name: "CCI14", type: "line", data: state.kline.map((x) => toNum(x.cci14)), xAxisIndex: 2, yAxisIndex: 2, symbol: "none", lineStyle: { width: 1 } });
  } else if (subIndicator === "wr") {
    series.push({ name: "WR14", type: "line", data: state.kline.map((x) => toNum(x.wr14)), xAxisIndex: 2, yAxisIndex: 2, symbol: "none", lineStyle: { width: 1 } });
  } else if (subIndicator === "atr") {
    series.push({ name: "ATR14", type: "line", data: state.kline.map((x) => toNum(x.atr14)), xAxisIndex: 2, yAxisIndex: 2, symbol: "none", lineStyle: { width: 1 } });
  }

  return {
    animation: false,
    legend: { top: 8, left: 8, type: "scroll" },
    tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    brush: {
      toolbox: ["lineX", "clear"],
      xAxisIndex: "all",
      brushMode: "single",
      throttleType: "debounce",
      throttleDelay: 300,
    },
    grid: [
      { left: 60, right: 24, top: 40, height: "55%" },
      { left: 60, right: 24, top: "68%", height: "12%" },
      { left: 60, right: 24, top: "83%", height: "12%" },
    ],
    xAxis: [
      { type: "category", data: dates, boundaryGap: true, axisLine: { onZero: false }, splitLine: { show: false } },
      { type: "category", data: dates, gridIndex: 1, boundaryGap: true, axisLabel: { show: false } },
      { type: "category", data: dates, gridIndex: 2, boundaryGap: true, axisLabel: { show: false } },
    ],
    yAxis: [
      { scale: true, splitArea: { show: true } },
      { scale: true, gridIndex: 1, splitNumber: 2 },
      { scale: true, gridIndex: 2, splitNumber: 2 },
    ],
    dataZoom: [
      { type: "inside", xAxisIndex: [0, 1, 2], start: state.dataZoom.start, end: state.dataZoom.end },
      { type: "slider", xAxisIndex: [0, 1, 2], bottom: 6, height: 18, start: state.dataZoom.start, end: state.dataZoom.end },
    ],
    series,
  };
}

function captureDataZoom() {
  const option = chart.getOption?.();
  const zoom = option?.dataZoom?.[0];
  if (!zoom) return;
  const start = Number(zoom.start);
  const end = Number(zoom.end);
  if (Number.isFinite(start) && Number.isFinite(end)) {
    state.dataZoom = { start, end };
  }
}

function renderChart(preserveZoom = true) {
  if (preserveZoom) captureDataZoom();
  chart.setOption(buildOption(), true);
}

function setLineMode(active) {
  state.lineMode = active;
  state.lineStartIndex = null;
  state.dragStartIndex = null;
  state.dragPreviewIndex = null;
  $("lineModeBtn").classList.toggle("activeTool", active);
  setStatus(active ? "手动画线：请在K线上按住鼠标拖出范围" : "");
  renderChart();
}

function updateIndicatorControls() {
  $("bollParamWrap").style.display = $("mainIndicator").value === "boll" ? "inline-flex" : "none";
}

function pointToDataIndex(offsetX, offsetY) {
  if (!state.kline.length) return null;
  const point = [offsetX, offsetY];
  if (!chart.containPixel({ gridIndex: 0 }, point)) return null;
  const coord = chart.convertFromPixel({ xAxisIndex: 0, yAxisIndex: 0 }, point);
  if (!coord || coord[0] === undefined || coord[0] === null) return null;
  const idx = Math.round(coord[0]);
  if (!Number.isFinite(idx)) return null;
  return Math.max(0, Math.min(state.kline.length - 1, idx));
}

function renderLabels() {
  const type = $("filterType").value;
  const freq = $("filterFreq").value;
  const list = state.labels.filter((x) => (!type || x.region_type === type) && (!freq || x.label_freq === freq));
  $("labelList").innerHTML = list.map((x) => `
    <div class="labelItem" data-id="${x.region_id}">
      <div class="id">${x.region_id}</div>
      <div class="meta">${x.start_date} ~ ${x.end_date}</div>
      <div class="meta">${labelZh("region_type", x.region_type)} / ${labelZh("cycle", x.cycle)} / ${labelZh("label_freq", x.label_freq)} / ${labelZh("level", x.level)} / 置信度${x.confidence} / ${labelZh("usable_for_signal", x.usable_for_signal)}</div>
      <div class="meta">${reasonsDisplay(x.reason)}</div>
    </div>
  `).join("");
  document.querySelectorAll(".labelItem").forEach((node) => {
    if (node.classList.contains("candidate")) return;
    node.addEventListener("click", () => openEdit(node.dataset.id));
  });
  renderCandidates();
}

function renderCandidates() {
  const type = $("filterType").value;
  const freq = $("filterFreq").value;
  const list = state.candidates.filter((x) => (!type || x.region_type === type) && (!freq || x.label_freq === freq));
  $("candidateCount").textContent = String(list.length);
  $("candidateList").innerHTML = list.map((x) => `
    <div class="labelItem candidate" data-id="${x.candidate_id}">
      <div class="labelTopLine">
        <div class="id">${x.candidate_id}</div>
        <button type="button" class="smallDanger candidateDeleteBtn" data-id="${x.candidate_id}">删除</button>
      </div>
      <div class="meta">${x.start_date} ~ ${x.end_date}</div>
      <div class="meta">${labelZh("region_type", x.region_type)} / ${labelZh("cycle", x.cycle)} / ${labelZh("label_freq", x.label_freq)} / ${labelZh("level", x.level)} / 分数=${x.score}</div>
      <div class="meta">${x.evidence || ""}</div>
    </div>
  `).join("");
  document.querySelectorAll(".labelItem.candidate").forEach((node) => {
    node.addEventListener("click", () => openCandidate(node.dataset.id));
  });
  document.querySelectorAll(".candidateDeleteBtn").forEach((node) => {
    node.addEventListener("click", async (event) => {
      event.stopPropagation();
      const candidateId = node.dataset.id;
      if (!candidateId) return;
      if (!confirm(`只删除候选 ${candidateId}，不会删除正式标注。确认删除？`)) return;
      try {
        await api(`/api/candidates/${encodeURIComponent(candidateId)}`, { method: "DELETE" });
        state.candidates = state.candidates.filter((x) => x.candidate_id !== candidateId);
        renderChart();
        renderLabels();
        setStatus(`已删除候选 ${candidateId}，正式标注未受影响`);
      } catch (err) {
        setStatus(err.message);
      }
    });
  });
}

async function loadAll() {
  setStatus("加载中...");
  state.lineStartIndex = null;
  const params = new URLSearchParams({ freq: state.freq });
  if ($("startDate").value) params.set("start_date", $("startDate").value);
  if ($("endDate").value) params.set("end_date", $("endDate").value);
  const [kline, labels, candidates] = await Promise.all([
    api(`/api/kline?${params}`),
    api("/api/labels"),
    api("/api/candidates"),
  ]);
  state.kline = kline.data || [];
  state.labels = labels.data || [];
  state.candidates = candidates.data || [];
  applyInitialLabelZoom();
  renderChart(false);
  renderLabels();
  setStatus(`K线 ${state.kline.length} 条，标注 ${state.labels.length} 条，候选 ${state.candidates.length} 条`);
}

async function loadMeta() {
  try {
    const meta = await api("/api/meta");
    state.meta = meta;
    if (meta.default_freq) {
      state.freq = meta.default_freq;
    }
    $("brandTitle").textContent = `${meta.index_name || meta.index_code} 顶底标注`;
    applyMetaConfig(meta);
  } catch (err) {
    setStatus(err.message);
  }
}

function applyMetaConfig(meta) {
  const allowed = new Set(meta.allowed_freqs || ["daily", "weekly", "monthly", "intraday"]);
  document.querySelectorAll("#freqButtons button[data-freq]").forEach((button) => {
    const visible = allowed.has(button.dataset.freq);
    button.style.display = visible ? "" : "none";
    button.classList.toggle("active", button.dataset.freq === state.freq);
  });
  if (!allowed.has(state.freq)) {
    state.freq = allowed.values().next().value || "daily";
  }
  const titleSuffix = meta.default_freq === "weekly" && allowed.size === 1 && allowed.has("weekly")
    ? "周线顶底标注"
    : "顶底标注";
  const title = `${meta.index_name || meta.index_code} ${titleSuffix}`;
  $("brandTitle").textContent = title;
  document.title = `${title}工具`;
  applyAllowedFreqOptions("filterFreq", allowed, true);
  applyAllowedFreqOptions("label_freq", allowed, false);
  applyAllowedCycleOptions(allowed);
  if (meta.default_freq === "weekly" && allowed.size === 1 && allowed.has("weekly")) {
    $("generateCandidatesBtn").style.display = "none";
    document.querySelector('label input#showCandidates')?.closest("label")?.style && (document.querySelector('label input#showCandidates').closest("label").style.display = "none");
    $("showCandidates").checked = false;
    document.querySelector(".panelHeader.compact").style.display = "none";
    $("candidateList").style.display = "none";
  }
}

function applyAllowedFreqOptions(selectId, allowed, keepAllOption) {
  const select = $(selectId);
  for (const option of select.options) {
    if (keepAllOption && option.value === "") {
      option.hidden = false;
      option.disabled = false;
      continue;
    }
    const visible = allowed.has(option.value);
    option.hidden = !visible;
    option.disabled = !visible;
  }
  if (select.value && !allowed.has(select.value)) {
    select.value = allowed.values().next().value || "";
  }
}

function applyAllowedCycleOptions(allowed) {
  const cycle = $("cycle");
  const intraday = [...cycle.options].find((option) => option.value === "intraday");
  if (!intraday) return;
  const visible = allowed.has("intraday");
  intraday.hidden = !visible;
  intraday.disabled = !visible;
  if (!visible && cycle.value === "intraday") {
    cycle.value = allowed.has("monthly") ? "long" : "medium";
  }
}

function formData() {
  const ids = ["region_id", "region_type", "form_start_date", "form_end_date", "cycle", "level", "label_freq", "confidence", "usable_for_signal", "entry_start", "entry_end", "exit_start", "exit_end", "reason", "notes"];
  const data = {};
  for (const id of ids) {
    const key = id === "form_start_date" ? "start_date" : id === "form_end_date" ? "end_date" : id;
    data[key] = $(id).value.trim();
  }
  return data;
}

function fillForm(item) {
  $("region_id").value = item.region_id || "";
  $("region_type").value = item.region_type || "bottom";
  $("form_start_date").value = item.start_date || "";
  $("form_end_date").value = item.end_date || "";
  $("cycle").value = item.cycle || "short";
  $("level").value = item.level || "swing";
  $("label_freq").value = item.label_freq || labelFreqFromChart();
  $("confidence").value = String(item.confidence || "2");
  $("usable_for_signal").value = String(item.usable_for_signal || "1");
  $("entry_start").value = item.entry_start || "";
  $("entry_end").value = item.entry_end || "";
  $("exit_start").value = item.exit_start || "";
  $("exit_end").value = item.exit_end || "";
  $("reason").value = item.reason || "";
  $("notes").value = item.notes || "";
  updateTypeFields();
  updateHint();
}

function openDialog(item, editingId = null) {
  state.editingId = editingId;
  $("dialogTitle").textContent = editingId ? "编辑标注" : "新增标注";
  $("deleteBtn").style.visibility = editingId ? "visible" : "hidden";
  fillForm(item);
  document.querySelector(".sidePanel").classList.add("editing");
}

function openNew(range = null) {
  const freq = labelFreqFromChart();
  const item = {
    region_type: "bottom",
    start_date: range?.start_date || "",
    end_date: range?.end_date || "",
    cycle: defaultCycle(freq),
    level: defaultLevel(freq),
    label_freq: freq,
    confidence: "2",
    usable_for_signal: "1",
    reason: "",
    notes: "",
  };
  item.entry_start = item.start_date;
  item.entry_end = item.end_date;
  item.exit_start = "";
  item.exit_end = "";
  item.region_id = nextRegionId(item);
  openDialog(item, null);
}

function openEdit(regionId) {
  const item = state.labels.find((x) => x.region_id === regionId);
  if (item) {
    zoomToItem(item);
    openDialog(item, regionId);
  }
}

function openCandidate(candidateId) {
  const candidate = state.candidates.find((x) => x.candidate_id === candidateId);
  if (!candidate) return;
  const item = {
    region_id: nextRegionId(candidate),
    start_date: candidate.start_date,
    end_date: candidate.end_date,
    region_type: candidate.region_type,
    cycle: candidate.cycle,
    level: candidate.level,
    label_freq: candidate.label_freq,
    confidence: "2",
    usable_for_signal: "1",
    entry_start: candidate.suggested_entry_start || "",
    entry_end: candidate.suggested_entry_end || "",
    exit_start: candidate.suggested_exit_start || "",
    exit_end: candidate.suggested_exit_end || "",
    reason: candidate.region_type === "bottom" ? "candidate_bottom" : "candidate_top",
    notes: `${candidate.notes || ""} ${candidate.evidence || ""}`.trim(),
  };
  openDialog(item, null);
}

function updateTypeFields() {
  const isBottom = $("region_type").value === "bottom";
  document.querySelectorAll(".bottomOnly").forEach((x) => x.style.display = isBottom ? "flex" : "none");
  document.querySelectorAll(".topOnly").forEach((x) => x.style.display = isBottom ? "none" : "flex");
  if (isBottom) {
    $("exit_start").value = "";
    $("exit_end").value = "";
    if (!$("entry_start").value) $("entry_start").value = $("form_start_date").value;
    if (!$("entry_end").value) $("entry_end").value = $("form_end_date").value;
  } else {
    $("entry_start").value = "";
    $("entry_end").value = "";
    if (!$("exit_start").value) $("exit_start").value = $("form_start_date").value;
    if (!$("exit_end").value) $("exit_end").value = $("form_end_date").value;
  }
}

function updateAutoId() {
  if (state.editingId) return;
  const item = formData();
  if (item.start_date && item.region_type && item.cycle && item.label_freq) {
    $("region_id").value = nextRegionId(item);
  }
}

function updateHint() {
  const item = formData();
  const hints = [];
  if (item.cycle === "intraday" && item.label_freq !== "intraday") hints.push("cycle=intraday 时 label_freq 必须为 intraday");
  if (item.label_freq === "intraday" && item.cycle !== "intraday") hints.push("label_freq=intraday 时 cycle 必须为 intraday");
  if (item.cycle === "short" && !["daily", "weekly"].includes(item.label_freq)) hints.push("short 推荐 daily，允许 weekly");
  if (item.cycle === "medium" && !["daily", "weekly"].includes(item.label_freq)) hints.push("medium 推荐 daily/weekly");
  if (item.cycle === "long" && !["weekly", "monthly"].includes(item.label_freq)) hints.push("long 推荐 weekly/monthly");
  $("formHint").textContent = hints.join("；");
}

async function saveLabel() {
  const data = formData();
  try {
    if (state.editingId) {
      await api(`/api/labels/${encodeURIComponent(state.editingId)}`, { method: "PUT", body: JSON.stringify(data) });
    } else {
      await api("/api/labels", { method: "POST", body: JSON.stringify(data) });
    }
    closeEditPane();
    await loadAll();
  } catch (err) {
    $("formHint").textContent = err.message;
  }
}

async function deleteLabel() {
  if (!state.editingId) return;
  if (!confirm(`确认删除 ${state.editingId}？`)) return;
  await api(`/api/labels/${encodeURIComponent(state.editingId)}`, { method: "DELETE" });
  closeEditPane();
  await loadAll();
}

function closeEditPane() {
  state.editingId = null;
  document.querySelector(".sidePanel").classList.remove("editing");
}

function initReasons() {
  $("reasonPreset").innerHTML += reasons.map((x) => `<option value="${x}">${reasonZh[x] || x}</option>`).join("");
  $("reasonPreset").addEventListener("change", () => {
    const v = $("reasonPreset").value;
    if (!v) return;
    const current = $("reason").value.split("|").map((x) => x.trim()).filter(Boolean);
    if (!current.includes(v)) current.push(v);
    $("reason").value = current.join("|");
    $("reasonPreset").value = "";
  });
}

function bindEvents() {
  $("freqButtons").addEventListener("click", (event) => {
    const btn = event.target.closest("button[data-freq]");
    if (!btn) return;
    document.querySelectorAll("#freqButtons button").forEach((x) => x.classList.remove("active"));
    btn.classList.add("active");
    state.freq = btn.dataset.freq;
    state.autoZoomPending = true;
    loadAll().catch((err) => setStatus(err.message));
  });
  $("reloadBtn").addEventListener("click", () => loadAll().catch((err) => setStatus(err.message)));
  $("lineModeBtn").addEventListener("click", () => setLineMode(!state.lineMode));
  $("generateCandidatesBtn").addEventListener("click", async () => {
    setStatus("正在生成候选顶底...");
    try {
      const result = await api("/api/candidates/regenerate", { method: "POST", body: "{}" });
      state.candidates = result.data || [];
      renderChart();
      renderLabels();
      setStatus(`候选已生成 ${state.candidates.length} 条：${result.path}`);
    } catch (err) {
      setStatus(err.message);
    }
  });
  $("showCandidates").addEventListener("change", renderChart);
  chart.on("dataZoom", () => {
    state.autoZoomPending = false;
    captureDataZoom();
  });
  $("exportBtn").addEventListener("click", async () => {
    const result = await api("/api/export");
    setStatus(`已导出/保存到 ${result.path}`);
  });
  $("newBtn").addEventListener("click", () => openNew());
  $("closeDialogBtn").addEventListener("click", closeEditPane);
  $("saveBtn").addEventListener("click", saveLabel);
  $("deleteBtn").addEventListener("click", deleteLabel);
  $("copyBtn").addEventListener("click", () => {
    state.editingId = null;
    $("dialogTitle").textContent = "复制为新周期";
    $("deleteBtn").style.visibility = "hidden";
    updateAutoId();
  });
  $("mainIndicator").addEventListener("change", () => {
    updateIndicatorControls();
    renderChart();
  });
  ["subIndicator", "bollParam"].forEach((id) => $(id).addEventListener("change", renderChart));
  ["filterType", "filterFreq"].forEach((id) => $(id).addEventListener("change", renderLabels));
  ["region_type", "form_start_date", "form_end_date"].forEach((id) => $(id).addEventListener("change", () => {
    updateTypeFields();
    updateAutoId();
  }));
  ["cycle", "label_freq", "level"].forEach((id) => $(id).addEventListener("change", () => {
    updateAutoId();
    updateHint();
  }));
  $("confidence").addEventListener("change", () => {
    $("usable_for_signal").value = $("confidence").value === "1" ? "0" : "1";
  });
  chart.on("brushSelected", (params) => {
    if (state.lineMode) return;
    const batch = params.batch?.[0];
    const selected = batch?.selected?.find((x) => x.seriesIndex === 0 && x.dataIndex?.length);
    if (!selected) return;
    const startIndex = Math.min(...selected.dataIndex);
    const endIndex = Math.max(...selected.dataIndex);
    if (!state.kline.length) return;
    state.selectedRange = normalizeRange(startIndex, endIndex);
    openNew(state.selectedRange);
    chart.dispatchAction({ type: "brush", areas: [] });
  });
  const zr = chart.getZr();
  zr.on("mousedown", (event) => {
    if (!state.lineMode) return;
    const index = pointToDataIndex(event.offsetX, event.offsetY);
    if (index === null) return;
    state.dragStartIndex = index;
    state.dragPreviewIndex = index;
    const row = state.kline[index];
    setStatus(`手动画线：起点 ${row.period_start || row.trade_date}，拖动到终点后松开`);
    renderChart();
  });
  zr.on("mousemove", (event) => {
    if (!state.lineMode || state.dragStartIndex === null) return;
    const index = pointToDataIndex(event.offsetX, event.offsetY);
    if (index === null || index === state.dragPreviewIndex) return;
    state.dragPreviewIndex = index;
    renderChart();
  });
  zr.on("mouseup", (event) => {
    if (!state.lineMode || state.dragStartIndex === null) return;
    const endIndex = pointToDataIndex(event.offsetX, event.offsetY);
    const startIndex = state.dragStartIndex;
    state.dragStartIndex = null;
    state.dragPreviewIndex = null;
    if (endIndex === null) {
      renderChart();
      return;
    }
    state.selectedRange = normalizeRange(startIndex, endIndex);
    setLineMode(false);
    openNew(state.selectedRange);
  });
  window.addEventListener("resize", () => chart.resize());
}

initReasons();
bindEvents();
updateIndicatorControls();
loadMeta().then(() => loadAll()).catch((err) => setStatus(err.message));
