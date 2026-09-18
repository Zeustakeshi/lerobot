# TurboVLA

TurboVLA is a vision-language-action policy for efficient robot manipulation. The
official implementation is maintained in the [TurboVLA repository](https://github.com/H-EmbodVis/TurboVLA).

> [!WARNING]
> This LeRobot integration is experimental. Strict conversion, strict loading, and
> fixed-input raw-action parity against the released LIBERO checkpoint are verified;
> published benchmark reproduction is not complete. LIBERO is the only release
> path validated by this integration. The RoboTwin configuration and converter
> path are not yet validated against a released RoboTwin checkpoint. Official
> released weights are simulation-trained. Do not treat them as real-robot safety
> or robustness claims.

## Installation

Install TurboVLA support as part of LeRobot:

```bash
pip install "lerobot[turbovla]"
```

For LIBERO evaluation dependencies, install both extras:

```bash
pip install "lerobot[turbovla,libero]"
```

No standalone `lerobot_policy_turbovla` package is required.

On first model construction, LeRobot downloads the configured DINOv3 and BERT
backbones from Hugging Face. DINOv3 requires accepting Meta's access terms. Set
`--policy.local_files_only=true` only after those artifacts have been cached, for
offline execution.

## Data and processor contract

The policy is registered as `turbovla`, so standard LeRobot configuration parsing
accepts `--policy.type=turbovla`. The validated LIBERO preset uses two ordered
cameras, 8-D state and 7-D actions, chunks of 12, DINOv3 ViT-B, and BERT base. A RoboTwin
schema is present for future compatibility, but is not a supported release path yet.

Camera tensors are ordered exactly as configured. They must contain RGB images in
channel-first `C,H,W` or batched `B,C,H,W` layout. `uint8` pixels are converted from
`[0,255]` to float32 `[0,1]`; floating input must already be in `[0,1]`. The default
LIBERO path follows upstream TurboVLA: the LIBERO environment adapter flips both image
axes, then the policy receives 256×256 RGB images normalized with the pinned DINO/ImageNet
mean and std. No StarVLA/OpenVLA-style
224px resize is applied. An optional deterministic center crop may be configured, but
the cropped image must still match the configured model size. LIBERO uses the agent-view
camera followed by eye-in-hand. RoboTwin uses head, left wrist, then right wrist.

The state vector and action vector use the benchmark's configured order and units;
processors do not infer or reorder joints. LIBERO state/proprio is normalized with
mean/std statistics, while actions use upstream min/max scaling to and from `[-1, 1]`.
Missing, incorrectly shaped, non-finite, or degenerate statistics are rejected. Policy
outputs are denormalized back to dataset action units before being returned.

At runtime, pass the current instruction as `task`. The processor validates its batch
size and exposes it to the policy as `observation.language`. Text is not rewritten:
TurboVLA's BERT base uncased tokenizer applies truncation at the configured maximum
length inside the text encoder. Empty or missing instructions fail explicitly.

The TurboVLA policy page records strict-conversion, vision-parity, A100 runtime,
model-latency, and LIBERO Spatial smoke-validation results. Evaluation fails before
rollout unless the resolved DINOv3 revision and its `preprocessor_config.json` match
the converted processor, including camera order, 180-degree orientation, rescaling,
and normalization.

## Benchmark fields to fill later

Leave these values blank in public artifacts until a full contract-compliant LIBERO
run has completed:

| Field                                  | Value |
| -------------------------------------- | ----- |
| Source checkpoint SHA256               |       |
| Source stats SHA256                    |       |
| Converted checkpoint revision          |       |
| LeRobot revision                       |       |
| TurboVLA upstream revision             |       |
| LIBERO revision                        |       |
| GPU name / driver / CUDA / PyTorch     |       |
| Precision                              |       |
| Seed                                   |       |
| Suites and trials per task             |       |
| Per-task success rates                 |       |
| Per-suite success rates                |       |
| Aggregate success rate and 95% CI      |       |
| Model-only latency mean / median / p95 |       |
| End-to-end latency mean / median / p95 |       |
| Peak allocated / reserved VRAM         |       |

## References and citation

- [Official TurboVLA source](https://github.com/H-EmbodVis/TurboVLA)

To convert a released upstream checkpoint into a portable LeRobot artifact, use
the bundled converter. Supply the exact source SHA256, variant, and statistics JSON:

```bash
lerobot-convert-turbovla \
  --source-path=<upstream-checkpoint> \
  --source-kind=<libero-pth|robotwin-ema-safetensors> \
  --source-url=<upstream-checkpoint-url-or-release-page> \
  --source-revision=<immutable-upstream-commit> \
  --checkpoint-license=<source-checkpoint-license> \
  --variant=<libero|robotwin> \
  --stats-json=<dataset-statistics.json> \
  --expected-sha256=<source-checkpoint-sha256> \
  --output-dir=<converted-checkpoint>
```

For public LIBERO artifacts, use `--source-kind=libero-pth` and `--variant=libero`.
The converter preserves source text-padding, vision model identity/revision, and attention metadata, then writes
`provenance.json` containing the source SHA256, benchmark variant, model identifiers,
and configured revisions. Publish that file with a
converted checkpoint. Add benchmark results only after reproducing them with the
documented evaluation command, checkpoint revision, hardware, and seed.

To run the released-checkpoint strict-conversion integration test locally, cache the
public BERT assets and set the source checkpoint path:

```bash
TURBOVLA_LIBERO_SOURCE_PATH=/path/to/turbovla_libero.pth \
TURBOVLA_LIBERO_SOURCE_SHA256=d031ad7be05a2f5d04afb3194ed26b0cb46083685edee7a5e145078a37d26bab \
uv run pytest tests/policies/turbovla/test_conversion_turbovla.py \
  -k released_libero_checkpoint_strict_conversion -sv
```

Use the citation supplied by the official TurboVLA project and review the licenses of
TurboVLA, DINOv3, BERT, the benchmark, and checkpoint parameters before redistribution.
