# ============================================================================
# Run FastAPI Backend Server
# ============================================================================

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "🚀 STARTING FASTAPI BACKEND SERVER" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment not activated!" -ForegroundColor Yellow
    Write-Host "Activating .venv..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

Write-Host ""
Write-Host "📡 Backend will be available at:" -ForegroundColor Green
Write-Host "   http://localhost:8000" -ForegroundColor White
Write-Host "   http://localhost:8000/docs (API Documentation)" -ForegroundColor White
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Set Python path
$env:PYTHONPATH = (Get-Location).Path

# Run uvicorn
uvicorn src.backend.app:app --host 0.0.0.0 --port 8000 --reload
