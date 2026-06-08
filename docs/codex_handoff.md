# Codex Handoff - stock-predict-000852

## 1. Project Goal

This project predicts the CSI 1000 index (`000852`) using market data and LLM-extracted news events.

Repository path:

`D:\python workbench\stock_predict\000852`

Current stage:

- The full data and modeling pipeline can run.
- Guba sentiment is temporarily disabled.
- The immediate goal is not live trading yet.
- The current focus is to evaluate whether model "upward" signals have statistical value.

Primary evaluation focus:

- When the model predicts "up", how often is the actual label also up?
- When the model assigns high upward probability, is future return positive?
- How do signal count, hit rate, average return, and median return change by probability threshold?
- Does the model beat simple baselines?

Do not over-focus on overall three-class accuracy. The trading-useful question is signal quality.

## 2. Git And Working Directory

GitHub repository:

`https://github.com/baogehuo2/stock-predict-000852.git`

Main branch:

`main`

When creating a new Codex conversation, choose the Git repository root:

`D:\python workbench\stock_predict\000852`

Do not choose the parent directory:

`D:\python workbench\stock_predict`

Otherwise Codex may not detect `.git`, and branch/worktree options may not appear.

Useful checks:

```powershell
cd "D:\python workbench\stock_predict\000852"
git status --short
git log --oneline -5
git remote -v
```

Important commits:

- `cebdd40` - Improve captcha handling and model feature pipeline
- `e565a26` - Disable guba sentiment features by config

Rules for future Codex sessions:

- Do not run `git reset --hard`.
- Do not revert user changes unless explicitly asked.
- Read `git status --short` before editing.

## 3. Current Data State

Confirmed data coverage:

`2016-01-01` to `2025-05-19`

Available data:

- Historical news data
- LLM-extracted news event data
- Index market data

Guba data exists or is being improved separately, but current model training does not use Guba sentiment features.

Historical news collection command previously used:

```powershell
python .\src\collectors\collect_news_history.py --start-date 2016-01-01 --end-date 2025-05-19 --keyword-mode all --sleep 1 --batch-days 10
```

Historical event extraction command:

```powershell
python -m src.llm.extract_event --start-date 2016-01-01 --end-date 2025-05-19 --limit-per-day 20 --batch-size 20 --llm-retries 3 --retry-wait 10
```

Avoid rerunning large LLM extraction unless intentionally filling missing data, because it consumes API calls and time.

## 4. Guba Sentiment Is Soft-Disabled

The user asked to temporarily turn off Guba sentiment analysis for modeling.

This was implemented as a soft disable, not deletion.

Config:

`config/config.yaml`

```yaml
features:
  use_guba_sentiment: false
```

Effects:

1. `src/features/build_model_dataset.py`
   - Does not aggregate from `sentiment_feature_daily`.
   - Does not write these features into `feature_json`:
     - `f_heat_score`
     - `f_heat_zscore_20d`
     - `f_sentiment_score`
     - `f_disagreement`

2. `src/modeling/train_lgbm.py`
   - Filters out old residual Guba sentiment features if an older dataset still contains them.

3. `main_daily_run.py`
   - Skips `build_sentiment_features` in the default flow when `features.use_guba_sentiment=false`.

Preserved:

- Guba raw collection code
- Guba raw database tables
- Guba sentiment feature code
- Existing Guba data

To re-enable later:

```yaml
features:
  use_guba_sentiment: true
```

Then rerun:

```powershell
python .\main_daily_run.py --step build_sentiment_features
python .\main_daily_run.py --step build_dataset
python .\main_daily_run.py --step train
```

## 5. Current Training Flow

If index data and LLM events are already in the database, run:

```powershell
python .\main_daily_run.py --step build_market_features
python .\main_daily_run.py --step build_dataset
python .\main_daily_run.py --step train
```

Do not run this while Guba sentiment is disabled:

```powershell
python .\main_daily_run.py --step build_sentiment_features
```

The recent near-term training command was:

```powershell
python -m src.modeling.train_lgbm --train-start 2016-01-01 --train-end 2023-12-31 --valid-start 2024-01-01 --valid-end 2024-12-31 --test-start 2025-01-01 --model-tag news2016_202505 --min-train-rows 200
```

Expected model artifacts:

- `models/lgbm_news2016_202505_3d.joblib`
- `models/lgbm_news2016_202505_5d.joblib`
- `models/lgbm_news2016_202505_7d.joblib`
- `models/metrics_news2016_202505.joblib`

If these files are missing on a different computer, rerun the training command.

## 6. Training Result Interpretation

Console label encoding can appear garbled:

- `涓婃定` = up
- `涓嬭穼` = down
- `闇囪崱` = flat/volatile

This is usually console encoding, not necessarily a model or database problem.

Recent near-term model `news2016_202505` approximate results:

3-day test:

- Overall accuracy: about `38.46%`
- Up pred_accuracy: about `50.00%`
- Up pred_count: about `96`
- Up actual_recall: about `34.04%`

5-day test:

- Overall accuracy: about `37.46%`
- Up pred_accuracy: about `50.00%`
- Up pred_count: about `38`
- Up actual_recall: about `13.38%`
- The model predicts flat/volatile too often.

7-day test:

- Overall accuracy: about `39.56%`
- Up pred_accuracy: about `55.00%`
- Up pred_count: about `80`
- Up actual_recall: about `27.85%`

Interpretation:

- Overall three-class accuracy is weak.
- The 7-day upward signal may have some value.
- The next step is to evaluate high-confidence upward probability signals.

## 7. Signal Quality Evaluation Script

New script:

`src/modeling/evaluate_signal_quality.py`

Purpose:

Evaluate whether upward model signals are useful for trading-style decisions.

It reports:

- `rule`
  - `baseline_all_rows`
  - `predicted_up_label`
  - `up_probability`
  - `predicted_up_label_year_YYYY`
- `threshold`
- `signal_count`
- `coverage`
- `actual_up_rate`
- `positive_ret_rate`
- `avg_future_ret`
- `median_future_ret`
- `avg_pred_ret`
- `avg_up_proba`

Evaluate the recent near-term model:

```powershell
python -m src.modeling.evaluate_signal_quality --model-tag news2016_202505 --test-start 2025-01-01 --thresholds 0.35,0.40,0.45,0.50,0.55,0.60 --output-csv data/reports/signal_quality_news2016_202505.csv
```

Evaluate default model:

```powershell
python -m src.modeling.evaluate_signal_quality --test-start 2025-01-01 --output-csv data/reports/signal_quality_default.csv
```

Important:

- This script was syntax-checked.
- `--help` was checked.
- Full local run could not complete in one Codex environment because local MySQL was not running.
- The user's machine with the database should be able to run it.

Decision rule after running:

- If high upward probability thresholds improve `actual_up_rate` and `avg_future_ret`, continue signal filtering/backtesting.
- If not, improve features and labels before tuning model parameters.

## 8. Recommended Next Optimization Steps

### Step 1: Run Signal Quality Evaluation

Run:

```powershell
python -m src.modeling.evaluate_signal_quality --model-tag news2016_202505 --test-start 2025-01-01 --thresholds 0.35,0.40,0.45,0.50,0.55,0.60 --output-csv data/reports/signal_quality_news2016_202505.csv
```

Inspect:

- `up_probability >= 0.45`
- `up_probability >= 0.50`
- `up_probability >= 0.55`
- `up_probability >= 0.60`

Useful signal means:

- enough `signal_count`
- `actual_up_rate` above baseline
- `positive_ret_rate` above baseline
- `avg_future_ret` positive

### Step 2: Add A Binary Buy Model

Current target is three-class:

- up
- down
- flat/volatile

Trading needs a binary decision:

- buy
- do not buy

Potential target:

```text
buy_label_5d = future_ret_5d > buy_threshold
```

Candidate thresholds:

- `future_ret_5d > 0`
- `future_ret_5d > 0.01`
- `future_ret_5d > 0.015`
- `future_ret_7d > 0.015`

Evaluate:

- buy precision
- buy recall
- buy signal count
- average future return
- yearly stability

### Step 3: Add Rolling Event Features

Current event features are single-day aggregates:

- `event_score`
- `event_count`
- group scores/counts

News impact often lasts several days. Add rolling features:

- last 1-day event score/count
- last 3-day event score/count
- last 5-day event score/count
- last 10-day event score/count
- rolling liquidity event strength
- rolling policy-market event strength
- rolling growth-industry event strength
- rolling macro-risk event strength
- rolling index-style event strength

This is likely more valuable than hyperparameter tuning.

### Step 4: Add Simple Baselines

Compare model against:

- always predict up
- always predict flat
- 20-day moving-average trend rule
- 5-day momentum positive rule
- 20-day return positive rule

If the model cannot beat simple baselines, improve labels/features before model tuning.

### Step 5: Fix Or Remove `f_amount_zscore_20d`

Training repeatedly warns:

```text
drop all-missing training features ['f_amount_zscore_20d']
```

This means the amount/turnover feature is empty in training.

Options:

1. Fix index amount data collection and feature generation.
2. Remove this feature from the feature list if amount data cannot be obtained.

Index volume/amount is important for timing, so fixing data is preferable.

## 9. Guba Captcha Work

Guba captcha was a separate improvement thread and can be postponed while Guba sentiment is disabled.

File:

`src/collectors/collect_guba_history_playwright.py`

Important implementation details:

- Eastmoney captcha uses sliced images, not a directly displayed full image.
- DOM uses:
  - `.em_cut_bg_slice`
  - `.em_cut_fullbg_slice`
  - `background-position`
- Code reconstructs captcha images using browser canvas.
- Debug files include:
  - `*_composed_bg.png`
  - `*_composed_full.png`
  - `*_download_piece.png`
  - `*_cv_diff.png`
  - `*_cv_threshold.png`
  - `*_cv_target_on_bg.png`
  - `*_cv_target_on_diff.png`
  - `*_cv_target_on_threshold.png`
  - `*_meta.json`

Captcha debug command:

```powershell
python .\src\collectors\collect_guba_history_playwright.py --bar-name 中证1000吧 --start-page 21 --end-page 22 --sleep 2 --detail-sleep 0.5 --fetch-comments --max-retries 6 --retry-wait 8 --captcha-solver cv --captcha-debug-dir data/debug/captcha
```

This can wait until the model/news path is better evaluated.

## 10. News Keyword Strategy

Historical collection should use:

```powershell
--keyword-mode all
```

Reason:

- Do not filter too aggressively at collection time.
- Classification and feature extraction can happen later.
- Missing historical data is harder to recover than noisy data.

Keyword groups:

- `index_style`
- `liquidity`
- `policy_market`
- `growth_industry`
- `macro`

News rows include:

- `matched_keywords`
- `matched_groups`

Modeling direction:

- daily direct CSI 1000 related news count
- daily/rolling liquidity news count/score
- daily/rolling policy-market news count/score
- daily/rolling growth-industry news count/score
- daily/rolling macro-risk news count/score

## 11. Common Commands

Build market features:

```powershell
python .\main_daily_run.py --step build_market_features
```

Build model dataset:

```powershell
python .\main_daily_run.py --step build_dataset
```

Default train:

```powershell
python .\main_daily_run.py --step train
```

Near-term train:

```powershell
python -m src.modeling.train_lgbm --train-start 2016-01-01 --train-end 2023-12-31 --valid-start 2024-01-01 --valid-end 2024-12-31 --test-start 2025-01-01 --model-tag news2016_202505 --min-train-rows 200
```

Signal quality evaluation:

```powershell
python -m src.modeling.evaluate_signal_quality --model-tag news2016_202505 --test-start 2025-01-01 --thresholds 0.35,0.40,0.45,0.50,0.55,0.60 --output-csv data/reports/signal_quality_news2016_202505.csv
```

## 12. New Codex Conversation Checklist

When a new Codex conversation starts, run:

```powershell
cd "D:\python workbench\stock_predict\000852"
git status --short
git log --oneline -5
python -m src.modeling.evaluate_signal_quality --help
```

Then ask the user to run, or run locally if the database is available:

```powershell
python -m src.modeling.evaluate_signal_quality --model-tag news2016_202505 --test-start 2025-01-01 --thresholds 0.35,0.40,0.45,0.50,0.55,0.60 --output-csv data/reports/signal_quality_news2016_202505.csv
```

After reviewing output:

- If high-threshold up signals improve hit rate and returns, build a simple signal backtest.
- If not, implement rolling event features and a binary buy model.

