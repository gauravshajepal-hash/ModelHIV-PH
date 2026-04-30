param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.tools\python312\python.exe'
if (-not (Test-Path $python)) {
    throw "Local Python runtime not found at $python. Bootstrap it first."
}

$src = Join-Path $repoRoot 'src'
$env:PYTHONPATH = $src
$env:PYTHONNOUSERSITE = '1'
& $python @PythonArgs
exit $LASTEXITCODE
