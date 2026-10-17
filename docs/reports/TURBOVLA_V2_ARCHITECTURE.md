# TurboVLA-v2 — architecture and LeRobot commands

## Scope

`turbovla_v2` is a new, standalone LeRobot policy. It does not inherit from or
import the existing `turbovla` implementation. TurboVLA was used only as the
behavioral reference for the expected LeRobot action-chunk interface.

The old DINOv3 vision tower and BERT text tower are replaced by the two aligned
towers from `google/tipsv2-b14`. The official checkpoint contains both towers
in one `model.safetensors`; its SentencePiece text tokenizer is stored in
`tokenizer.model`. The local `.cache/tipsv2-b14` directory already contains
both files, so there is no separate text-only artifact to download.

Primary sources:

- Paper: <https://arxiv.org/abs/2604.12012>
- Official code: <https://github.com/google-deepmind/tips>
- Official checkpoint: <https://huggingface.co/google/tipsv2-b14>

## Architecture

```text
RGB views [0,1], 252x252               instruction, <=64 tokens
             │                                  │
       TIPSv2-B/14 vision                  SentencePiece
       dense patch tokens                 TIPSv2 text tower
             │  [B,V,324,768]                   │ [B,768]
             └──────── projection to 384 ───────┘
                              │
                  + projected robot state
                              │
                4-layer multimodal encoder
                              │
            32 learned action queries (default)
                              │
                 3-layer cross-attention decoder
                              │
                     action chunk [B,32,A]
```

TIPSv2 receives RGB in `[0,1]` without ImageNet normalization. The default
252×252 input is divisible by patch size 14 and stays close to TurboVLA's
256×256 input. Dense patch tokens preserve the spatial signal while the global
TIPSv2 text embedding supplies language aligned to the same pretraining space.
The state token, multimodal fusion layers, action queries, decoder, and action
head are trained for the target LeRobot dataset.

By default the pretrained text tower is frozen and the vision tower is
trainable. Set `--policy.freeze_tips_vision=true` when GPU memory is limited.

## Install

```bash
conda activate lerobot_arm
pip install -e '.[turbovla-v2]'
```

To use the already downloaded offline checkpoint, add these options to train:

```text
--policy.tips_model_id=.cache/tipsv2-b14
--policy.local_files_only=true
```

## Train

The camera keys, state width, and action width are inferred from dataset
metadata and saved in the checkpoint.

```bash
conda activate lerobot_arm
lerobot-train \
  --policy.type=turbovla_v2 \
  --policy.device=cuda \
  --policy.precision=bfloat16 \
  --policy.tips_model_id=.cache/tipsv2-b14 \
  --policy.local_files_only=true \
  --policy.chunk_size=32 \
  --policy.n_action_steps=4 \
  --dataset.repo_id=YOUR_ORG/YOUR_DATASET \
  --dataset.root=/path/to/your/dataset \
  --output_dir=outputs/turbovla_v2 \
  --job_name=turbovla_v2 \
  --batch_size=8 \
  --steps=20000
```

## Rollout with LeRobot

Use the camera configuration and robot identity used during recording. The task
string is consumed by the TIPSv2 text tower.

```bash
conda activate lerobot_arm
lerobot-rollout \
  --strategy.type=base \
  --inference.type=sync \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=YOUR_ROBOT_ID \
  --robot.cameras='YOUR_CAMERA_CONFIG' \
  --policy.path=outputs/turbovla_v2/checkpoints/last/pretrained_model \
  --dataset.single_task='pick up the target object' \
  --fps=30
```

The command is syntactically accepted by the current `lerobot-rollout` CLI.
Physical rollout still requires matching robot calibration, camera keys, state
and action dimensions, and a trained checkpoint. Run the robot initially at a
safe speed with an emergency stop available.

## Verification performed

- LeRobot registry resolves `turbovla_v2` and its policy class.
- `lerobot-train --help` and `lerobot-rollout --help` expose the new policy.
- The local pretrained TIPSv2 checkpoint loads in `lerobot_arm`.
- A real TIPSv2 vision + text forward pass, action loss backward pass, and
  action-chunk prediction completed on CPU.
