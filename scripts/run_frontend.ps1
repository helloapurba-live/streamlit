# ============================================================================
# Run Streamlit Frontend Dashboard
# ============================================================================

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "🎨 STARTING STREAMLIT DASHBOARD" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan

if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment not activated!" -ForegroundColor Yellow
    Write-Host "Activating .venv..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

Write-Host ""
Write-Host "📊 Dashboard will be available at:" -ForegroundColor Green
Write-Host "   http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""

$env:PYTHONPATH = (Get-Location).Path

streamlit run src\frontend\app.py --server.port 8501
