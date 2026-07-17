# D3 Integration

v0.8.1 provides an optional, standalone D3 learned-detector adapter. It implements the D3 second-order temporal-feature computation with a documented single-video preprocessing adaptation. It does not claim full upstream runtime equivalence until actual pretrained encoder inference and upstream model parity are executed in the local environment.

## 1. Method Overview

D3, Detection by Difference of Differences, computes first-order distances between consecutive frame embeddings, then computes second-order differences between those first-order values. The native raw score is the standard deviation of the second-order sequence.

## 2. Pinned Upstream Source

- Repository: https://github.com/Zig-HS/D3
- Pinned commit: `c798fbc57fe0c4198d63a73732c2c0f9e4b4816c`
- Commit date: `2026-06-23T16:44:02+08:00`
- Paper: https://arxiv.org/abs/2508.00701

## 3. License

The upstream D3 project is MIT licensed.

## 4. Official Upstream Workflow

Upstream extracts a 3 second clip at 8 FPS into frame folders with ffmpeg, builds CSV files for real and fake folders, loads frames through OpenCV, applies crop/resize/normalization, runs `D3_model`, and evaluates AP with `average_precision_score(1-y_true, y_pred)`.

## 5. Local Single-Video Adaptation

This project analyzes one selected local video. It decodes frames directly with OpenCV instead of creating the upstream dataset folder/CSV workflow. D3 output remains separate under `learned_detector_results` and is not fused into `unified_evidence`, `evidence_timeline.jsonl`, or `ai_interpretation_input.json`.

## 6. Exact Preprocessing

See `docs/D3_UPSTREAM_PARITY.md` for the comparison table. The local path uses the same window duration, seed logic, frame-count policy, crop, resize target, normalization constants, dtype, and tensor layout. It deliberately differs in decode backend and converts BGR to RGB before normalization.

## 7. Exact Mathematical Computation

For embeddings shaped `[b, t, d]`:

- L2: `torch.norm(vec1 - vec2, p=2, dim=-1)`
- Cosine: `F.cosine_similarity(vec1, vec2, dim=-1)`
- Second order: `first[:, 1:] - first[:, :-1]`
- Mean: `torch.mean(second, dim=1)`
- Native score: `torch.std(second, dim=1)`

## 8. Standard-Deviation Convention

Upstream uses `torch.std` without overriding correction. The local summary uses sample standard deviation to match PyTorch default semantics for deterministic math tests.

## 9. Encoder Mappings

The authoritative local mapping is `learned_detectors/d3/configuration.py::ENCODER_REGISTRY`.

- `CLIP-16`: `openai/clip-vit-base-patch16`, `transformers.CLIPVisionModel`
- `CLIP-32`: `openai/clip-vit-base-patch32`, `transformers.CLIPVisionModel`
- `XCLIP-16`: `microsoft/xclip-base-patch16`, `transformers.XCLIPVisionModel`
- `XCLIP-32`: `microsoft/xclip-base-patch32`, `transformers.XCLIPVisionModel`
- `DINO-base`: `facebook/dinov2-base`, `transformers.AutoModel`
- `DINO-large`: `facebook/dinov2-large`, `transformers.AutoModel`
- `ResNet-18`: `torchvision/resnet18`
- `VGG-16`: `torchvision/vgg16`
- `EfficientNet-b4`: `torchvision/efficientnet_b4`
- `MobileNet-v3`: `mobilenetv3_large_100`, `timm.create_model`

All encoders are currently `implemented_unverified` locally until actual runtime parity is executed.

## 10. Score Direction

Score direction is not verified. Upstream labels real as `0`, fake as `1`, then evaluates AP with `1-y_true`, making real the AP positive class under the CSV labels. Because this conflicts with intuitive generated-artifact wording and no threshold is supplied, the local report keeps `score_direction` as `not_verified`.

## 11. What the Score Means

`raw_score` is the uncalibrated standard deviation of the second-order temporal feature sequence for the selected D3 window.

## 12. What the Score Does Not Mean

It is not a probability, confidence, thresholded label, fake/real verdict, authenticity verdict, manipulation verdict, or risk score.

## 13. Configuration

- `LEARNED_DETECTORS_ENABLED`: default `false`
- `D3_ENABLED`: default `false`
- `D3_DEVICE`: `auto`, `cpu`, `cuda`; default `auto`
- `D3_ENCODER`: default `XCLIP-16`
- `D3_DISTANCE`: `l2`, `cos`; default `l2`
- `D3_RANDOM_SEED`: default `42`
- `D3_TIMEOUT_SECONDS`: positive integer seconds; default `300`
- `D3_ALLOW_MODEL_DOWNLOAD`: default `false`
- `D3_MODEL_CACHE_DIRECTORY`: optional cache path
- `D3_PREPROCESSING_MODE`: currently `upstream_compatible`
- `D3_PRESERVE_TEMPORARY_FRAMES`: default `false`

## 14. Device Handling

`auto` selects CUDA when available, otherwise CPU. Explicit `cuda` returns a structured failure when CUDA is unavailable.

## 15. Timeout Handling

The heavy D3 path runs in a worker process. If it exceeds `D3_TIMEOUT_SECONDS`, the process is terminated and the result status is `timed_out` with reason `detector_timeout`.

## 16. Model Asset Setup

Install optional dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-d3.txt
```

Prepare assets explicitly by allowing downloads for a controlled run:

```powershell
$env:D3_ALLOW_MODEL_DOWNLOAD="true"
$env:D3_MODEL_CACHE_DIRECTORY="<local-cache-directory>"
```

Then run D3 once. Later offline runs should set `D3_ALLOW_MODEL_DOWNLOAD=false` and reuse the cache.

## 17. Offline Behavior

When downloads are disabled, Hugging Face loaders receive `local_files_only=True`. torchvision/timm pretrained downloads are avoided by not requesting pretrained weights. Missing dependencies or assets must produce `unavailable` or `failed` results, not silent downloads.

## 18. Temporary-Frame Behavior

Default `false`: no diagnostic D3 frames are retained.  
When `true`: frames are written under `d3_temporary_frames/` inside the analysis directory and the relative directory is recorded in preprocessing metadata.

## 19. Artifact Schemas

Completed D3 runs write:

- `d3_detector_result.json`
- `d3_temporal_features.jsonl`

The JSONL records distinguish `d3_first_order_feature` and `d3_second_order_feature`. They do not contain embeddings, probabilities, confidences, or classifications.

## 20. Execution Statuses

See `docs/LEARNED_DETECTORS.md` for central status and reason-code semantics.

## 21. Actual Tested Environment

Current local venv has OpenCV but does not have `torch`, `torchvision`, `transformers`, or `timm`. Actual pretrained XCLIP-16 inference did not run in this environment.

## 22. Upstream Parity Results

Pinned source was inspected. Mathematical parity is tested with deterministic tensors. Runtime parity is not verified because optional ML dependencies and model assets are absent.

## 23. Known Deviations

- Single-video workflow instead of upstream dataset CSV workflow.
- OpenCV timestamp seeks instead of upstream ffmpeg frame extraction.
- RGB conversion before normalization, while upstream code reads BGR images and does not explicitly convert.
- No actual pretrained runtime parity result yet.

## 24. Limitations

D3 remains a standalone raw-feature detector. It has no local calibration, no operating threshold, no score direction claim, and no verdict layer.

## 25. v0.8.2 Deferred Work

v0.8.2 may add full-video multi-window D3 analysis, timeline mapping, forensic/D3 alignment, agreement/conflict analysis, and updated final evidence bundle support. None of that is implemented in v0.8.1.
