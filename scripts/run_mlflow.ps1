# ============================================================================
# Run MLflow Tracking UI
# ============================================================================

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "🧪 STARTING MLFLOW TRACKING UI" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan

if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment not activated!" -ForegroundColor Yellow
    Write-Host "Activating .venv..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

Write-Host ""
Write-Host "📊 MLflow UI will be available at:" -ForegroundColor Green
Write-Host "   http://localhost:5000" -ForegroundColor White
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""

mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
