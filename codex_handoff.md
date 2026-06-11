# Buy Signal Dashboard Handoff

## Branch

- Current branch: `codex/buy-signal-dashboard`
- Base branch: `feature/buy-signal-research`
- This branch contains the uncommitted Buy signal visualization work.
- Do not merge or commit this work from `feature/buy-signal-research` directly.

## Background

Model 1.0 has been frozen as a 7-day binary Buy model:

- `model_version=buy-signal-v1.0`
- `feature_version=buy-features-v1.0`
- Positive label: `future_ret_7d > 0.5%`
- Signal score threshold: raw `buy_proba >= 0.60`
- Hard causal regime gate: only the causal bull state may emit `long`
- `buy_proba` is a relative model score, not a calibrated success probability
- `future_ret_7d` is an ex-post realized return, not information available on the prediction date

The user requested a standalone HTML dashboard for the frozen model output. This dashboard is visualization only. It does not retrain the model, regenerate predictions, execute trades, or perform a formal backtest.

## Implemented Function

Run:

```powershell
python .\main_daily_run.py --step report_buy_signal
```

Output:

```text
data/reports/buy_signal_v1_dashboard.html
```

The report reads `buy_signal_daily` and `market_index_daily` from MySQL and displays:

1. Latest direction, causal regime, Buy score, and threshold.
2. Buy score versus signal threshold.
3. CSI 1000 close prices with Long markers.
4. Realized 7-day returns for Long signals.
5. Realized Long precision and average future return.
6. Recent signal details and pending-verification status.

## Files Changed

- `src/report/generate_buy_signal_report.py`: standalone HTML dashboard generator.
- `main_daily_run.py`: registers the `report_buy_signal` step.
- `README.md`: documents the command and output path.
- `codex_handoff.md`: this handoff record.

## Verification Completed

The following checks passed:

```powershell
python -m py_compile .\src\report\generate_buy_signal_report.py .\main_daily_run.py
git diff --check
python .\main_daily_run.py --step report_buy_signal --stop-on-error
```

The command generated the HTML successfully. At verification time, `buy_signal_daily` contained only one Model 1.0 signal dated `2026-05-15`, so the dashboard correctly showed limited history and no realized 7-day result. Once data and signals are updated through `2026-06-10`, rerunning the same command will expand the charts automatically.

## Remaining Work

- Review the dashboard after current data and signals have been updated.
- Commit and push only from `codex/buy-signal-dashboard` when the user confirms.
- Keep formal backtesting outside this branch.

