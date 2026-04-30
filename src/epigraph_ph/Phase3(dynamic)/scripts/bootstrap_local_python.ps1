param(
    [string]$SourcePython = 'C:\Users\gaura\AppData\Local\Programs\Python\Python312'
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $repoRoot '.tools\python312'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$null = robocopy $SourcePython $dest /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
if ($LASTEXITCODE -ge 8) {
    throw "robocopy failed with exit code $LASTEXITCODE"
}
$python = Join-Path $dest 'python.exe'
if (-not (Test-Path $python)) {
    throw "python.exe was not copied to $python"
}
Write-Host "Bootstrapped local Python runtime to $dest"
