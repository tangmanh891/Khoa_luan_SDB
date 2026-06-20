param(
    [ValidateSet("pdf")]
    [string]$Target = "pdf",
    [switch]$Release
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Thesis = Join-Path $Root "publications\thesis"
$Build = Join-Path $Thesis "build"
$Releases = Join-Path $Thesis "releases"

if (Test-Path $Build) {
    Remove-Item -Recurse -Force $Build | Out-Null
}
New-Item -ItemType Directory -Force -Path $Build, $Releases | Out-Null

# Dọn dẹp các tệp phụ trợ cũ tại thư mục gốc của thesis nếu tồn tại
$LegacyFiles = @("main.aux", "main.log", "main.out", "main.toc", "main.pdf", "main.bbl", "main.bcf", "main.blg", "main.run.xml")
foreach ($file in $LegacyFiles) {
    $path = Join-Path $Thesis $file
    if (Test-Path $path) {
        Remove-Item -Force $path | Out-Null
    }
}

& python (Join-Path $PSScriptRoot "sync_experimental_results.py") --check
if ($LASTEXITCODE -ne 0) {
    throw "Experimental result outputs are stale. Run sync_experimental_results.py --write."
}

if ($Target -eq "pdf") {
    Push-Location $Thesis
    try {
        & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) { throw "First pdflatex pass failed." }

        & biber --input-directory=build --output-directory=build main
        if ($LASTEXITCODE -ne 0) { throw "Biber pass failed." }

        & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) { throw "Second pdflatex pass failed." }

        & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) { throw "Third pdflatex pass failed." }

        & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) { throw "Final pdflatex pass failed." }
    }
    finally {
        Pop-Location
    }

    if ($Release) {
        Copy-Item -Force -LiteralPath (Join-Path $Build "main.pdf") `
            -Destination (Join-Path $Releases "AutoShotV2_Thesis.pdf")
    }
}

Write-Host "Build completed: target=$Target release=$Release"
