param(
    [ValidateSet("", "report_buy_signal", "report_walk_forward_buy")]
    [string]$Step = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($Step -eq "") {
    python .\run_buy_signal_dashboard.py
} else {
    python .\run_buy_signal_dashboard.py --step $Step
}

