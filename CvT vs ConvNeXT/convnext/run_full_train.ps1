$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "train_convnextv2_food101.py"
$dataRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\food-101"))
$output = Join-Path $PSScriptRoot "outputs\convnextv2_tiny_food101"

New-Item -ItemType Directory -Force $output | Out-Null

& $python $script `
  --data-root $dataRoot `
  --output-dir $output `
  --model-name "convnextv2_tiny" `
  --epochs 20 `
  --batch-size 8 `
  --accum-steps 2 `
  --num-workers 4
