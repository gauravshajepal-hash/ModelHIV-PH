param(
    [string]$Distro = "Ubuntu",
    [string]$RepoPath = "D:\EpiGraph_PH",
    [string]$VenvPath = "/home/gaurav/.venvs/modelhiv-ph-gpu"
)

function Convert-ToWslPath {
    param([string]$WindowsPath)
    $resolved = (Resolve-Path $WindowsPath).Path
    $drive = $resolved.Substring(0, 1).ToLowerInvariant()
    $suffix = $resolved.Substring(2).Replace("\", "/")
    return "/mnt/$drive$suffix"
}

$repoWsl = Convert-ToWslPath -WindowsPath $RepoPath
$scriptWsl = "$repoWsl/scripts/bootstrap_wsl_gpu_env.sh"
$escapedVenv = $VenvPath.Replace("'", "'\"'\"'")
$escapedRepo = $repoWsl.Replace("'", "'\"'\"'")
$escapedScript = $scriptWsl.Replace("'", "'\"'\"'")

wsl -d $Distro bash -lc "chmod +x '$escapedScript' && EPIGRAPH_WSL_VENV='$escapedVenv' '$escapedScript' '$escapedRepo'"
