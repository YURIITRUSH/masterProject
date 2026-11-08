$scenarios = @(
    @{ file = "scenario.txt";        tag = "scenario" },
    @{ file = "scenario-stress.txt"; tag = "stress"   }
)
$port = "COM5"

foreach ($s in $scenarios) {
    $scen = $s.file
    $name = $s.tag
    
    Write-Host "`n--- Сценарій: $scen (tag: $name) ---" -ForegroundColor Cyan
    
    Write-Host "-> Baseline..." -ForegroundColor Yellow
    python main.py collect --scenario $scen --port $port --tag "base_$name"
    
    Write-Host "-> Hybrid..." -ForegroundColor Yellow
    python main.py ai --scenario $scen --static --port $port --tag "hybrid_$name"
    
    Write-Host "-> Dynamic AI..." -ForegroundColor Yellow
    python main.py ai --scenario $scen --port $port --tag "ai_$name"
}