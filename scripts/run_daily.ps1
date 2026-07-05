param(
    [ValidateSet("", "report_buy_signal", "report_walk_forward_buy")]
    [string]$Step = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($Step -eq "") {
    python .\main_daily_run.py
} else {
    python .\main_daily_run.py --step $Step
}

