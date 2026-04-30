param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot 'run_local_python.ps1'
& $runner '-m' 'phase3_dynamic.cli' @CliArgs
exit $LASTEXITCODE
