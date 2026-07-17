# D3 Upstream Parity Record

Pinned upstream repository: https://github.com/Zig-HS/D3  
Pinned commit: `c798fbc57fe0c4198d63a73732c2c0f9e4b4816c`  
Commit date: `2026-06-23T16:44:02+08:00`  
License: MIT  
Paper: https://arxiv.org/abs/2508.00701

## Files Inspected

- `README.md`
- `LICENSE`
- `requirements.txt`
- `eval.py`
- `models/D3_model.py`
- `data/datasets.py`
- `utils/video2frame.py`
- `utils/folder2csv.py`

## Preprocessing Comparison

| Component | Upstream behavior | Local behavior | Status | Notes |
|---|---|---|---|---|
| Window duration | 3 seconds | 3 seconds | exact | Same duration. |
| Start time | `0` when duration <= 3, otherwise `floor(random.uniform(0, duration - 3))` with seed 42 in utility main | Same formula with local `random.Random(seed)` | numerically_equivalent | Local avoids global RNG mutation. |
| Requested frame rate | ffmpeg `fps=8` | OpenCV timestamp seeks at 8 FPS | deliberate_adaptation | Decoding backend differs. |
| Frame naming | `%d.jpg` from ffmpeg | In-memory frames; optional diagnostic names | deliberate_adaptation | Normal analysis does not require persisted frame folder. |
| Frame ordering | Numeric filename sort | Timestamp order from generated list | numerically_equivalent | Equivalent for successfully decoded frames. |
| Minimum frame count | Fails below 8 frames | Returns `skipped/insufficient_frames` below 8 | deliberate_adaptation | Local preserves main report. |
| 8 vs 16 selection | 8 when fewer than 16, else 16 | Same | exact | Tested. |
| Crop | Remove `int(0.1 * longer_dimension)` from both sides of longer dimension | Same | exact | Tested. |
| Resize | Albumentations `Resize(224, 224)` | OpenCV `INTER_LINEAR` resize to 224 x 224 | numerically_equivalent | Albumentations uses OpenCV resize by default. |
| Color order | `cv2.imread` BGR into Albumentations Normalize | OpenCV BGR converted to RGB before ImageNet normalization | mismatch | Upstream code does not explicitly convert BGR to RGB. Local uses RGB intentionally because metadata says RGB. |
| Normalization | ImageNet mean/std, max pixel 255 | Same values | exact | Tested finite output. |
| Tensor layout | `[t, 3, 224, 224]`, DataLoader gives batch `[b, t, 3, h, w]` | `[1, t, 3, 224, 224]` | exact | Same model input shape. |
| Decode failures | Missing frames reduce folder count; dataset raises if total < 8 | Failed seeks are skipped; skipped if valid frames < 8 | deliberate_adaptation | Local records structured status. |

## Mathematical Comparison

| Component | Upstream behavior | Local behavior | Status | Notes |
|---|---|---|---|---|
| Encoder output | Reshape encoder output to `[b, t, -1]` | Same | exact | Runtime parity not executed here. |
| L2 first order | `torch.norm(vec1 - vec2, p=2, dim=-1)` | Same formula in Python test helper and adapter model | exact | Synthetic tests pass. |
| Cosine first order | `F.cosine_similarity(vec1, vec2, dim=-1)` | Same formula | exact | Synthetic tests pass. |
| Second order | `dis_1st[:, 1:] - dis_1st[:, :-1]` | Same | exact | Synthetic tests pass. |
| Mean | `torch.mean(dis_2nd, dim=1)` | Same conceptual mean | exact | Synthetic tests pass. |
| Standard deviation | `torch.std(dis_2nd, dim=1)` default unbiased sample std | Local summary uses sample std for parity | exact | Fixed from v0.8 population std. |
| Native score | second-order std | `native_d3_second_order_standard_deviation` raw score | exact_math_not_runtime_verified | Requires actual model assets for full runtime parity. |

Synthetic mathematical tolerance: absolute `1e-7`, relative `1e-6`.

## Score Direction Evidence

`utils/folder2csv.py` assigns real videos label `0` and AI videos label `1`.

`eval.py` appends `batch_dis_std` to `y_pred`, appends dataset labels to `y_true`, then computes `average_precision_score(1-y_true, y_pred)`.

Conclusion: the upstream AP target is `1 - label`, which makes real videos the positive AP class under the upstream CSV labels. Because the paper motivation discusses generated-video temporal artifacts and no threshold/calibration path is provided in code, this integration keeps `score_direction` as `not_verified` and exposes no real/fake direction.

## Runtime Parity

Runtime parity is `not_verified` in this environment. The local venv is missing `torch`, `torchvision`, `transformers`, and `timm`; pretrained XCLIP-16 assets were not loaded.

## Final Parity Conclusion

v0.8.1 verifies pinned source, documents preprocessing adaptations, and verifies D3 math with deterministic synthetic tests. It does not claim full upstream runtime equivalence until actual pretrained inference and upstream model comparison are executed.
