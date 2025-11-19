# ============================================================================
# Run Complete ML Training Pipeline
# ============================================================================

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "🚀 RUNNING COMPLETE ML PIPELINE" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan

if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment not activated!" -ForegroundColor Yellow
    Write-Host "Activating .venv..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

$env:PYTHONPATH = (Get-Location).Path

Write-Host ""
Write-Host "Step 1/4: Generating synthetic data..." -ForegroundColor Yellow
python data\generate_synthetic_transactions.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Data generation failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2/4: Training scikit-learn Random Forest..." -ForegroundColor Yellow
python src\ml\train_sklearn.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Sklearn training failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 3/4: Training PyTorch + Skorch MLP..." -ForegroundColor Yellow
python src\ml\train_pytorch.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ PyTorch training failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 4/4: Hyperparameter tuning with Optuna..." -ForegroundColor Yellow
python src\ml\optuna_tune.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Optuna tuning encountered issues (non-critical)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "="*70 -ForegroundColor Green
Write-Host "✅ PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Green
Write-Host ""
Write-Host "Models saved to: models\" -ForegroundColor Cyan
Write-Host "MLflow experiments logged to: mlruns\" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run MLflow UI: .\scripts\run_mlflow.ps1" -ForegroundColor White
Write-Host "  2. Start backend: .\scripts\run_backend.ps1" -ForegroundColor White
Write-Host "  3. Start frontend: .\scripts\run_frontend.ps1" -ForegroundColor White
