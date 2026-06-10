param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectSlug,

    [string]$PythonVersion = "3.11",

    [switch]$SetupVenv,

    [switch]$Force
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
$templateDir = Join-Path $repoRoot "templates"
$projectsDir = Join-Path $repoRoot "projects"
$projectDir = Join-Path $projectsDir $ProjectSlug
$requirementsFile = Join-Path $repoRoot "requirements.txt"

if (-not (Test-Path $templateDir)) {
    throw "Template directory not found: $templateDir"
}

if (Test-Path $projectDir) {
    if (-not $Force) {
        throw "Project already exists: $projectDir. Re-run with -Force to overwrite."
    }
    Remove-Item -Recurse -Force $projectDir
}

New-Item -ItemType Directory -Path $projectDir -Force | Out-Null
Copy-Item -Path (Join-Path $templateDir "*") -Destination $projectDir -Recurse -Force

# Ensure runtime folders expected by structure.md exist.
$requiredDirs = @(
    (Join-Path $projectDir "data\raw"),
    (Join-Path $projectDir "data\interim"),
    (Join-Path $projectDir "data\processed"),
    (Join-Path $projectDir "runs")
)

foreach ($dir in $requiredDirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

# Replace obvious placeholders in copied markdown files.
$readmePath = Join-Path $projectDir "README.md"
if (Test-Path $readmePath) {
    $readmeText = Get-Content -Path $readmePath -Raw
    $readmeText = $readmeText -replace "\{\{PROJECT_NAME\}\}", $ProjectSlug
    Set-Content -Path $readmePath -Value $readmeText
}

$programPath = Join-Path $projectDir "program.md"
if (Test-Path $programPath) {
    $programText = Get-Content -Path $programPath -Raw
    $programText = $programText -replace "\{\{PROJECT_NAME\}\}", $ProjectSlug
    Set-Content -Path $programPath -Value $programText
}

if ($SetupVenv) {
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) {
        Write-Warning "uv is not installed. Scaffold completed, environment setup skipped."
    }
    else {
        Push-Location $projectDir
        try {
            & uv venv --python $PythonVersion .venv
            if (Test-Path $requirementsFile) {
                & uv pip install -r $requirementsFile
            }
            else {
                Write-Warning "requirements.txt not found at repo root."
            }
        }
        finally {
            Pop-Location
        }
    }
}

Write-Host "Scaffold completed:" $projectDir
Write-Host "Next steps:"
Write-Host "  1) Fill docs/00_problem_statement.md and docs/01_data_card.md"
Write-Host "  2) Confirm docs/03_cv_strategy.md"
Write-Host "  3) Run: python src/train.py --config configs/baseline.yaml"
