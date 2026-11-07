$scenarios = @("scenario.txt", "scenario-stress.txt")
$port = "COM5"

foreach ($scen in $scenarios) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($scen)
    Write-Host "`n--- Сценарій: $name ---" -ForegroundColor Cyan

    Write-Host "-> Baseline..." -ForegroundColor Yellow
    python main.py collect --scenario $scen --port $port --tag "base_$name"
    
    Write-Host "-> Hybrid..." -ForegroundColor Yellow
    python main.py ai --target 15.0 --static --port $port --tag "hybrid_$name"
    
    Write-Host "-> Dynamic AI..." -ForegroundColor Yellow
    python main.py ai --target 15.0 --port $port --tag "ai_$name"
}