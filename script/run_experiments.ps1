# run_experiments.ps1
# 依次训练、评估、可视化 resnet18 和 resnet34，最后输出对比表格

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   FCN 语义分割对比实验" -ForegroundColor Cyan
Write-Host "   resnet18 vs resnet34" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$Backbones = @("resnet18", "resnet34")

foreach ($backbone in $Backbones) {
    Write-Host ">>> [$backbone] 开始训练..." -ForegroundColor Yellow
    python train.py --backbone $backbone
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!!! [$backbone] 训练失败，退出码: $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }

    Write-Host ">>> [$backbone] 开始评估..." -ForegroundColor Yellow
    python evaluate.py --backbone $backbone
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!!! [$backbone] 评估失败，退出码: $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }

    Write-Host ">>> [$backbone] 开始可视化..." -ForegroundColor Yellow
    python visualize.py --backbone $backbone --num_samples 3
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!!! [$backbone] 可视化失败，退出码: $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }

    Write-Host ">>> [$backbone] 完成！`n" -ForegroundColor Green
}

Write-Host ">>> 生成对比表格..." -ForegroundColor Yellow
python compare_experiments.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "!!! compare_experiments.py 失败，退出码: $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   全部实验完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
