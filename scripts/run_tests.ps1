# ============================================================================
# Run Test Suite
# ============================================================================

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "🧪 RUNNING TEST SUITE" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan

if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment not activated!" -ForegroundColor Yellow
    Write-Host "Activating .venv..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

$env:PYTHONPATH = (Get-Location).Path

Write-Host ""
Write-Host "Running pytest..." -ForegroundColor Yellow
Write-Host ""

pytest tests\ -v --tb=short --cov=src --cov-report=term-missing

Write-Host ""
Write-Host "✅ Tests complete!" -ForegroundColor Green
