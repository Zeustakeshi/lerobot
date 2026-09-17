# TurboVLA

TurboVLA is a vision-language-action policy for efficient robot manipulation. The
official implementation is maintained in the [TurboVLA repository](https://github.com/H-EmbodVis/TurboVLA).

> [!WARNING]
> This LeRobot integration is experimental. Phase 2 provides native discovery,
> configuration, processor, and checkpoint-format tracers; faithful model inference
> and published benchmark parity are not implemented yet. Official released weights
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

## Configuration tracer

The policy is registered as `turbovla`, so standard LeRobot configuration parsing
accepts `--policy.type=turbovla`. The LIBERO preset uses two ordered cameras, 7-D
state/actions, chunks of 12, DINOv3 ViT-B, and BERT base. The RoboTwin schema uses
three ordered cameras, 14-D state/actions, chunks of 50, and DINOv3 ViT-L.

Model execution intentionally raises a clear `NotImplementedError` until the faithful
network port is complete. No success-rate, latency, VRAM, or numerical-parity result is
claimed by this integration phase.

## References and citation

- [Official TurboVLA source](https://github.com/H-EmbodVis/TurboVLA)
- [LeRobot integration plan](../../plans/plan.md)

Use the citation supplied by the official TurboVLA project and review the licenses of
TurboVLA, DINOv3, BERT, the benchmark, and checkpoint parameters before redistribution.
