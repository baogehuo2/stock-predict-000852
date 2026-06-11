@echo off
setlocal EnableExtensions

cd /d "%~dp0.."

set "LOOKBACK_DAYS=15"
if /I "%~1"=="--help" goto :help
if not "%~1"=="" set "LOOKBACK_DAYS=%~1"
echo(%LOOKBACK_DAYS%| findstr /r "^[1-9][0-9]*$" >nul
if errorlevel 1 goto :invalid_days

for /f %%D in ('powershell -NoProfile -Command "(Get-Date).AddDays(-%LOOKBACK_DAYS%).ToString('yyyy-MM-dd')"') do set "START_DATE=%%D"
for /f %%D in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd')"') do set "END_DATE=%%D"

if not defined START_DATE goto :date_failed
if not defined END_DATE goto :date_failed

echo ============================================================
echo Buy Signal Model 1.0 Daily Update
echo Project: %CD%
echo Window : %START_DATE% to %END_DATE%
echo Note   : Existing database rows are upserted or skipped.
echo ============================================================

call :step "1/6 Collect index history"
python -m src.collectors.collect_index_akshare --start-date %START_DATE% --end-date %END_DATE%
if errorlevel 1 goto :failed

call :step "2/6 Collect historical news"
python -m src.collectors.collect_news_history --start-date %START_DATE% --end-date %END_DATE% --source cctv --source baidu_economic --keyword-mode all
if errorlevel 1 goto :failed

call :step "3/6 Extract events from newly added news"
python -m src.llm.extract_event --start-date %START_DATE% --end-date %END_DATE% --limit-per-day 20 --batch-size 20 --fail-fast
if errorlevel 1 goto :failed

call :step "4/6 Rebuild market features"
python .\main_daily_run.py --step build_market_features --stop-on-error
if errorlevel 1 goto :failed

call :step "5/6 Rebuild model dataset"
python .\main_daily_run.py --step build_dataset --stop-on-error
if errorlevel 1 goto :failed

call :step "6/6 Generate official Buy signals"
python -m src.modeling.generate_buy_signal --start-date %START_DATE% --end-date %END_DATE% --output-csv data/reports/buy_signal_v1_recent.csv
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo Daily update completed successfully.
echo Signal CSV: %CD%\data\reports\buy_signal_v1_recent.csv
echo ============================================================
pause
exit /b 0

:step
echo.
echo ------------------------------------------------------------
echo [STEP] %~1
echo ------------------------------------------------------------
exit /b 0

:date_failed
echo Failed to calculate the rolling date window.
pause
exit /b 1

:invalid_days
echo LOOKBACK_DAYS must be a positive integer.
echo Example: scripts\run_buy_signal_daily.bat 30
pause
exit /b 1

:failed
echo.
echo ============================================================
echo Daily update stopped because a step failed.
echo Window: %START_DATE% to %END_DATE%
echo Fix the error, then run this BAT again. Completed rows will not
echo be duplicated because collectors and signals use database upsert.
echo ============================================================
pause
exit /b 1

:help
echo Usage:
echo   scripts\run_buy_signal_daily.bat
echo   scripts\run_buy_signal_daily.bat LOOKBACK_DAYS
echo.
echo Default LOOKBACK_DAYS is 15 calendar days.
exit /b 0
