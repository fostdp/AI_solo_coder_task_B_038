# 微服务启动脚本 - Windows PowerShell
# 生物制药冻干机微服务架构

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " 生物制药冻干机微服务架构 - 启动脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$services = @(
    @{ Name = "profinet_driver"; Port = "None"; Color = "Green" },
    @{ Name = "temp_controller"; Port = "None"; Color = "Yellow" },
    @{ Name = "quality_predictor"; Port = "None"; Color = "Cyan" },
    @{ Name = "alarm_publisher"; Port = "None"; Color = "Red" },
    @{ Name = "db_writer"; Port = "None"; Color = "Magenta" },
    @{ Name = "endpoint_detector"; Port = "None"; Color = "DarkCyan" },
    @{ Name = "defrost_optimizer"; Port = "None"; Color = "DarkYellow" },
    @{ Name = "fleet_controller"; Port = "None"; Color = "DarkMagenta" },
    @{ Name = "defect_detector"; Port = "None"; Color = "DarkRed" },
    @{ Name = "api_gateway"; Port = "8000"; Color = "Blue" }
)

$processes = @()

try {
    foreach ($service in $services) {
        $serviceDir = Join-Path $baseDir $service.Name
        $mainFile = Join-Path $serviceDir "main.py"
        
        if (-not (Test-Path $mainFile)) {
            Write-Host "[WARN] $($service.Name) 主文件不存在，跳过" -ForegroundColor Yellow
            continue
        }

        Write-Host "[START] 启动 $($service.Name)..." -ForegroundColor $service.Color
        
        $process = Start-Process -FilePath "python" `
            -ArgumentList "main.py" `
            -WorkingDirectory $serviceDir `
            -PassThru `
            -NoNewWindow
        
        $processes += @{ Process = $process; Service = $service.Name }
        
        Write-Host "[OK] $($service.Name) 已启动 (PID: $($process.Id))" -ForegroundColor Green
        
        Start-Sleep -Seconds 2
    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " 所有微服务启动完成!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "服务列表:" -ForegroundColor Cyan
    foreach ($proc in $processes) {
        Write-Host "  - $($proc.Service) (PID: $($proc.Process.Id))" -ForegroundColor White
    }
    Write-Host ""
    Write-Host "API Gateway: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "API Docs:    http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "按 Ctrl+C 停止所有服务..." -ForegroundColor Yellow

    try {
        while ($true) {
            Start-Sleep -Seconds 1
            
            foreach ($proc in $processes) {
                if ($proc.Process.HasExited) {
                    Write-Host "[WARN] $($proc.Service) 已退出 (Exit Code: $($proc.Process.ExitCode))" -ForegroundColor Yellow
                }
            }
        }
    }
    finally {
        Write-Host ""
        Write-Host "正在停止所有服务..." -ForegroundColor Yellow
        
        foreach ($proc in $processes) {
            if (-not $proc.Process.HasExited) {
                try {
                    Stop-Process -Id $proc.Process.Id -Force -ErrorAction SilentlyContinue
                    Write-Host "[STOP] $($proc.Service) 已停止" -ForegroundColor Gray
                }
                catch {
                    Write-Host "[ERROR] 停止 $($proc.Service) 失败: $_" -ForegroundColor Red
                }
            }
        }
        
        Write-Host ""
        Write-Host "所有服务已停止" -ForegroundColor Green
    }
}
catch {
    Write-Host "[ERROR] 启动失败: $_" -ForegroundColor Red
    
    foreach ($proc in $processes) {
        if (-not $proc.Process.HasExited) {
            try {
                Stop-Process -Id $proc.Process.Id -Force -ErrorAction SilentlyContinue
            }
            catch { }
        }
    }
    
    exit 1
}
