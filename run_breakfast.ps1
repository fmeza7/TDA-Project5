param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,

    [string]$OutputRoot = "pipeline\pipeline2_tda_dl\artifacts_breakfast",
    [string]$CameraFolder = "cam01",
    [string]$AnnotationSuffixes = ".labels",
    [string]$SplitFile = "pipeline\pipeline2_tda_dl\configs\breakfast_subject_split_34_9_9.json",
    [string]$TrainSubjects = "",
    [string]$ValSubjects = "",
    [string]$TestSubjects = "",
    [ValidateSet("frames", "seconds", "auto")]
    [string]$TimeUnits = "frames",
    [double]$SampleFps = 3.0,
    [int]$SmoothWindow = 5,
    [int]$WindowSize = 31,
    [int]$StrideTrain = 5,
    [int]$StrideVal = 5,
    [int]$StrideTest = 1,
    [int]$Epochs = 30,
    [int]$BatchSize = 32,
    [double]$LearningRate = 1e-3,
    [int]$KernelSize = 5,
    [double]$MinSegmentSec = 0.5,
    [switch]$LowStorageMode,
    [switch]$KeepTrainValWindows,
    [switch]$InstallDependencies,
    [switch]$StrictAnnotations
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoPython {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return (Resolve-Path -LiteralPath $venvPython).Path
    }
    return "python"
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    Write-Host ($Arguments -join " ")
    & $script:PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo en paso: $Name"
    }
}

function Remove-IfExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathToRemove
    )

    if (Test-Path -LiteralPath $PathToRemove) {
        Write-Host "Removing $PathToRemove" -ForegroundColor Yellow
        Remove-Item -LiteralPath $PathToRemove -Force
    }
}

$resolvedDatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
$resolvedOutputRoot = Join-Path $PSScriptRoot $OutputRoot
$PythonExe = Resolve-RepoPython
$resolvedSplitFile = $null

if ($SplitFile) {
    $candidateSplitPath = $SplitFile
    if (-not [System.IO.Path]::IsPathRooted($candidateSplitPath)) {
        $candidateSplitPath = Join-Path $PSScriptRoot $candidateSplitPath
    }
    if (Test-Path -LiteralPath $candidateSplitPath) {
        $resolvedSplitFile = (Resolve-Path -LiteralPath $candidateSplitPath).Path
    }
    else {
        throw "No existe SplitFile: $candidateSplitPath"
    }
}

if (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
    throw "No se encontro Python. Crea .venv o instala Python en PATH."
}

Write-Host "Repo root: $PSScriptRoot"
Write-Host "Dataset root: $resolvedDatasetRoot"
Write-Host "Output root: $resolvedOutputRoot"
Write-Host "Camera folder: $CameraFolder"
Write-Host "Python: $PythonExe"
Write-Host "Low storage mode: $LowStorageMode"
if ($resolvedSplitFile) {
    Write-Host "Split file: $resolvedSplitFile"
}

if ($InstallDependencies) {
    Write-Host ""
    Write-Host "=== Install Dependencies ===" -ForegroundColor Cyan
    & $PythonExe -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudieron instalar dependencias."
    }
}

$manifestPath = Join-Path $resolvedOutputRoot "manifest.json"
$cubicalDir = Join-Path $resolvedOutputRoot "outputs_cubical"
$curvesDir = Join-Path $resolvedOutputRoot "outputs_curves"
$frameLabelsDir = Join-Path $resolvedOutputRoot "frame_labels"
$windowsDir = Join-Path $resolvedOutputRoot "windows"
$temporalModelDir = Join-Path $resolvedOutputRoot "temporal_model"
$inferenceDir = Join-Path $resolvedOutputRoot "inference"
$decodedDir = Join-Path $resolvedOutputRoot "decoded"
$evalPath = Join-Path $resolvedOutputRoot "eval\eval_test.json"

$strictArgs = @()
if ($StrictAnnotations) {
    $strictArgs += "--strict_annotations"
}

$manifestArgs = @(
    "-m", "pipeline.pipeline2_tda_dl.breakfast_manifest_builder",
    "--videos_dir", $resolvedDatasetRoot,
    "--annotations_dir", $resolvedDatasetRoot,
    "--output_manifest", $manifestPath,
    "--camera_folders", $CameraFolder,
    "--annotation_suffixes", $AnnotationSuffixes
)
if ($resolvedSplitFile) {
    $manifestArgs += @("--split_file", $resolvedSplitFile)
}
if ($TrainSubjects) {
    $manifestArgs += @("--train_subjects", $TrainSubjects)
}
if ($ValSubjects) {
    $manifestArgs += @("--val_subjects", $ValSubjects)
}
if ($TestSubjects) {
    $manifestArgs += @("--test_subjects", $TestSubjects)
}
$manifestArgs += $strictArgs

Invoke-Step -Name "Build Manifest" -Arguments $manifestArgs

Invoke-Step -Name "Cubical Preprocessing" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.breakfast_cubical_preprocessing",
    "--dataset_manifest", $manifestPath,
    "--output_dir", $cubicalDir,
    "--sample_fps", "$SampleFps"
)

Invoke-Step -Name "Topological Curves" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.breakfast_curves",
    "--input_dir", $cubicalDir,
    "--output_dir", $curvesDir,
    "--smooth_window", "$SmoothWindow"
)

Invoke-Step -Name "Build Frame Labels" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.build_frame_labels",
    "--dataset_manifest", $manifestPath,
    "--cubical_manifest", (Join-Path $cubicalDir "manifest_cubical.json"),
    "--output_dir", $frameLabelsDir,
    "--train_split_names", "train",
    "--time_units", $TimeUnits
)

if ($LowStorageMode) {
    Remove-IfExists -PathToRemove (Join-Path $windowsDir "train_windows.npz")
    Remove-IfExists -PathToRemove (Join-Path $windowsDir "val_windows.npz")
    Remove-IfExists -PathToRemove (Join-Path $windowsDir "test_windows.npz")

    Invoke-Step -Name "Build Temporal Windows (Train+Val)" -Arguments @(
        "-m", "pipeline.pipeline2_tda_dl.breakfast_temporal_windows",
        "--cubical_manifest", (Join-Path $cubicalDir "manifest_cubical.json"),
        "--curves_manifest", (Join-Path $curvesDir "manifest_curves.json"),
        "--frame_labels_manifest", (Join-Path $frameLabelsDir "manifest_frame_labels.json"),
        "--output_dir", $windowsDir,
        "--window_size", "$WindowSize",
        "--stride_train", "$StrideTrain",
        "--stride_val", "$StrideVal",
        "--stride_test", "$StrideTest",
        "--splits", "train,val"
    )
}
else {
    Invoke-Step -Name "Build Temporal Windows" -Arguments @(
        "-m", "pipeline.pipeline2_tda_dl.breakfast_temporal_windows",
        "--cubical_manifest", (Join-Path $cubicalDir "manifest_cubical.json"),
        "--curves_manifest", (Join-Path $curvesDir "manifest_curves.json"),
        "--frame_labels_manifest", (Join-Path $frameLabelsDir "manifest_frame_labels.json"),
        "--output_dir", $windowsDir,
        "--window_size", "$WindowSize",
        "--stride_train", "$StrideTrain",
        "--stride_val", "$StrideVal",
        "--stride_test", "$StrideTest"
    )
}

Invoke-Step -Name "Train Temporal Segmenter" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.train_breakfast_temporal_segmenter",
    "--windows_dir", $windowsDir,
    "--output_dir", $temporalModelDir,
    "--epochs", "$Epochs",
    "--batch_size", "$BatchSize",
    "--lr", "$LearningRate",
    "--ignore_unknown",
    "--class_weighting"
)

if ($LowStorageMode -and -not $KeepTrainValWindows) {
    Remove-IfExists -PathToRemove (Join-Path $windowsDir "train_windows.npz")
    Remove-IfExists -PathToRemove (Join-Path $windowsDir "val_windows.npz")
}

if ($LowStorageMode) {
    Remove-IfExists -PathToRemove (Join-Path $windowsDir "test_windows.npz")
    Invoke-Step -Name "Build Temporal Windows (Test)" -Arguments @(
        "-m", "pipeline.pipeline2_tda_dl.breakfast_temporal_windows",
        "--cubical_manifest", (Join-Path $cubicalDir "manifest_cubical.json"),
        "--curves_manifest", (Join-Path $curvesDir "manifest_curves.json"),
        "--frame_labels_manifest", (Join-Path $frameLabelsDir "manifest_frame_labels.json"),
        "--output_dir", $windowsDir,
        "--window_size", "$WindowSize",
        "--stride_train", "$StrideTrain",
        "--stride_val", "$StrideVal",
        "--stride_test", "$StrideTest",
        "--splits", "test"
    )
}

Invoke-Step -Name "Infer Temporal Segmenter" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.infer_breakfast_temporal_segmenter",
    "--windows_npz", (Join-Path $windowsDir "test_windows.npz"),
    "--model_checkpoint", (Join-Path $temporalModelDir "breakfast_temporal_best.pt"),
    "--frame_labels_manifest", (Join-Path $frameLabelsDir "manifest_frame_labels.json"),
    "--output_dir", $inferenceDir
)

Invoke-Step -Name "Decode Predictions" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.decode_breakfast_predictions",
    "--raw_manifest", (Join-Path $inferenceDir "raw_predictions_manifest.json"),
    "--output_dir", $decodedDir,
    "--kernel_size", "$KernelSize",
    "--min_segment_sec", "$MinSegmentSec"
)

Invoke-Step -Name "Evaluate Segmentation" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.eval_breakfast_segmentation",
    "--decoded_manifest", (Join-Path $decodedDir "decoded_manifest.json"),
    "--frame_labels_manifest", (Join-Path $frameLabelsDir "manifest_frame_labels.json"),
    "--splits", "test",
    "--output_json", $evalPath
)

Write-Host ""
Write-Host "Pipeline completo." -ForegroundColor Green
Write-Host "Eval report: $evalPath"
