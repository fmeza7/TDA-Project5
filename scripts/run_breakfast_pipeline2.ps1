[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$VideosDir,

    [Parameter(Mandatory = $true)]
    [string]$AnnotationsDir,

    [Parameter(Mandatory = $true)]
    [string]$ArtifactsDir,

    [string]$SplitFile = "",
    [string]$TrainSubjects = "",
    [string]$ValSubjects = "",
    [string]$TestSubjects = "",
    [string]$CameraDirs = "cam01",
    [ValidateSet("auto", "seconds", "frames")]
    [string]$TimeUnits = "frames",
    [double]$SampleFps = 3.0,
    [int]$WindowSize = 31,
    [int]$StrideTrain = 5,
    [int]$StrideVal = 5,
    [int]$StrideTest = 1,
    [int]$Epochs = 30,
    [int]$BatchSize = 32,
    [double]$LearningRate = 1e-3,
    [int]$Seed = 1337,
    [double]$MinSegmentSec = 0.5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    Write-Host ("python " + ($Arguments -join " ")) -ForegroundColor DarkGray
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo la etapa: $Title"
    }
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

$resolvedVideosDir = (Resolve-Path $VideosDir).Path
$resolvedAnnotationsDir = (Resolve-Path $AnnotationsDir).Path

if (-not $SplitFile -and (-not $TrainSubjects -or -not $ValSubjects -or -not $TestSubjects)) {
    throw "Debes pasar -SplitFile o bien -TrainSubjects, -ValSubjects y -TestSubjects."
}

New-Item -ItemType Directory -Force -Path $ArtifactsDir | Out-Null
$resolvedArtifactsDir = (Resolve-Path $ArtifactsDir).Path

$manifestPath = Join-Path $resolvedArtifactsDir "manifest.json"
$cubicalDir = Join-Path $resolvedArtifactsDir "outputs_cubical"
$cubicalManifest = Join-Path $cubicalDir "manifest_cubical.json"
$curvesDir = Join-Path $resolvedArtifactsDir "outputs_curves"
$curvesManifest = Join-Path $curvesDir "manifest_curves.json"
$frameLabelsDir = Join-Path $resolvedArtifactsDir "frame_labels"
$frameLabelsManifest = Join-Path $frameLabelsDir "manifest_frame_labels.json"
$windowsDir = Join-Path $resolvedArtifactsDir "windows"
$temporalModelDir = Join-Path $resolvedArtifactsDir "temporal_model"
$modelCheckpoint = Join-Path $temporalModelDir "breakfast_temporal_best.pt"
$inferenceDir = Join-Path $resolvedArtifactsDir "inference"
$rawManifest = Join-Path $inferenceDir "raw_predictions_manifest.json"
$decodedDir = Join-Path $resolvedArtifactsDir "decoded"
$decodedManifest = Join-Path $decodedDir "decoded_manifest.json"
$evalDir = Join-Path $resolvedArtifactsDir "eval"
$evalJson = Join-Path $evalDir "eval_test.json"

$manifestArgs = @(
    "-m", "pipeline.pipeline2_tda_dl.breakfast_manifest_builder",
    "--videos_dir", $resolvedVideosDir,
    "--annotations_dir", $resolvedAnnotationsDir,
    "--output_manifest", $manifestPath,
    "--camera_dirs", $CameraDirs
)

if ($SplitFile) {
    $manifestArgs += @("--split_file", (Resolve-Path $SplitFile).Path)
}
else {
    $manifestArgs += @(
        "--train_subjects", $TrainSubjects,
        "--val_subjects", $ValSubjects,
        "--test_subjects", $TestSubjects
    )
}

Invoke-Step -Title "1/9 Manifest" -Arguments $manifestArgs

Invoke-Step -Title "2/9 Cubical Preprocessing" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.breakfast_cubical_preprocessing",
    "--dataset_manifest", $manifestPath,
    "--output_dir", $cubicalDir,
    "--sample_fps", "$SampleFps"
)

Invoke-Step -Title "3/9 Topological Curves" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.breakfast_curves",
    "--input_dir", $cubicalDir,
    "--output_dir", $curvesDir,
    "--smooth_window", "5"
)

Invoke-Step -Title "4/9 Frame Labels" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.build_frame_labels",
    "--dataset_manifest", $manifestPath,
    "--cubical_manifest", $cubicalManifest,
    "--output_dir", $frameLabelsDir,
    "--train_split_names", "train",
    "--time_units", $TimeUnits
)

Invoke-Step -Title "5/9 Temporal Windows" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.breakfast_temporal_windows",
    "--cubical_manifest", $cubicalManifest,
    "--curves_manifest", $curvesManifest,
    "--frame_labels_manifest", $frameLabelsManifest,
    "--output_dir", $windowsDir,
    "--window_size", "$WindowSize",
    "--stride_train", "$StrideTrain",
    "--stride_val", "$StrideVal",
    "--stride_test", "$StrideTest"
)

Invoke-Step -Title "6/9 Train Temporal Segmenter" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.train_breakfast_temporal_segmenter",
    "--windows_dir", $windowsDir,
    "--output_dir", $temporalModelDir,
    "--epochs", "$Epochs",
    "--batch_size", "$BatchSize",
    "--lr", "$LearningRate",
    "--seed", "$Seed",
    "--ignore_unknown",
    "--class_weighting"
)

Invoke-Step -Title "7/9 Inference" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.infer_breakfast_temporal_segmenter",
    "--windows_npz", (Join-Path $windowsDir "test_windows.npz"),
    "--model_checkpoint", $modelCheckpoint,
    "--frame_labels_manifest", $frameLabelsManifest,
    "--output_dir", $inferenceDir,
    "--seed", "$Seed"
)

Invoke-Step -Title "8/9 Decode Predictions" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.decode_breakfast_predictions",
    "--raw_manifest", $rawManifest,
    "--output_dir", $decodedDir,
    "--kernel_size", "5",
    "--min_segment_sec", "$MinSegmentSec"
)

Invoke-Step -Title "9/9 Evaluate" -Arguments @(
    "-m", "pipeline.pipeline2_tda_dl.eval_breakfast_segmentation",
    "--decoded_manifest", $decodedManifest,
    "--frame_labels_manifest", $frameLabelsManifest,
    "--splits", "test",
    "--output_json", $evalJson
)

Write-Host ""
Write-Host "Pipeline completo." -ForegroundColor Green
Write-Host "Manifest: $manifestPath"
Write-Host "Modelo:   $modelCheckpoint"
Write-Host "Eval:     $evalJson"
