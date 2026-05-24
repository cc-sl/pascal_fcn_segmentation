$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

$Backbones = "resnet18", "resnet34"

foreach ($backbone in $Backbones) {
    python train.py --backbone $backbone
    if ($LASTEXITCODE -ne 0) { exit 1 }

    python evaluate.py --backbone $backbone
    if ($LASTEXITCODE -ne 0) { exit 1 }

    python visualize.py --backbone $backbone --num_samples 3
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
