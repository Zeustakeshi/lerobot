# TurboVLA

TurboVLA is a vision-language-action policy for efficient robot manipulation. The
official implementation is maintained in the [TurboVLA repository](https://github.com/H-EmbodVis/TurboVLA).

> [!WARNING]
> This LeRobot integration is experimental. Native model execution and the LIBERO
> processor contract are implemented, but converted-checkpoint numerical parity and
> published benchmark reproduction are not complete. Official released weights
> are simulation-trained. Do not treat them as real-robot safety or robustness claims.

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

## Data and processor contract

The policy is registered as `turbovla`, so standard LeRobot configuration parsing
accepts `--policy.type=turbovla`. The LIBERO preset uses two ordered cameras, 7-D
state/actions, chunks of 12, DINOv3 ViT-B, and BERT base. The RoboTwin schema uses
three ordered cameras, 14-D state/actions, chunks of 50, and DINOv3 ViT-L.

Camera tensors are ordered exactly as configured. They must contain RGB images in
channel-first `C,H,W` or batched `B,C,H,W` layout. `uint8` pixels are converted from
`[0,255]` to float32 `[0,1]`; floating input must already be in `[0,1]`. The default
LIBERO path follows upstream TurboVLA: images must already be pre-rotated and sized to
256×256, then normalized with the DINO/ImageNet mean and std. No StarVLA/OpenVLA-style
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

No success-rate, latency, VRAM, or converted-checkpoint numerical-parity result is
claimed yet.

## References and citation

- [Official TurboVLA source](https://github.com/H-EmbodVis/TurboVLA)
- [LeRobot integration plan](../../plans/plan.md)

Use the citation supplied by the official TurboVLA project and review the licenses of
TurboVLA, DINOv3, BERT, the benchmark, and checkpoint parameters before redistribution.
