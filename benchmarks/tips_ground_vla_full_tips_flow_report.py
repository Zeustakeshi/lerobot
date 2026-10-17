"""Create expert-action alignment figures and the focused final report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "docs/reports/assets/tips_ground_vla_full_tips_flow"
JOINTS = ["Shoulder pan", "Shoulder lift", "Elbow flex", "Wrist flex", "Wrist roll", "Gripper"]
COLORS = {"Expert": "#1d2433", "Full TIPS Flow": "#2864dc", "Historical Direct": "#687386"}


def load_results(root: Path, seed: int):
    run = root / f"seed{seed}"
    flow = torch.load(run / "selected_checkpoint.pt", map_location="cpu", weights_only=False)
    direct = torch.load(root / "historical_direct_on_final_validation.pt", map_location="cpu", weights_only=False)
    return run, flow, direct


def setup_axis(ax, title: str | None = None):
    ax.axvspan(-0.35, 3.35, color="#f4ce55", alpha=0.22, zorder=0)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.set_xlim(-0.5, 31.5)
    if title:
        ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def get_examples(flow: dict, max_examples: int = 6) -> list[tuple[str, int]]:
    motion = np.asarray(flow["sample_motion"])
    events = flow["sample_events"]
    chosen = [int(motion.argmin()), int(np.argsort(motion)[len(motion) // 2]), int(motion.argmax())]
    reversal = max(range(len(events)), key=lambda i: events[i]["reversals"])
    gripper = max(range(len(events)), key=lambda i: events[i]["gripper_motion"])
    stopping = max(range(len(events)), key=lambda i: events[i]["stops"])
    chosen.extend([reversal, gripper, stopping])
    unique = list(dict.fromkeys(chosen))[:max_examples]
    labels = []
    for index in unique:
        if index == stopping and index == chosen[0]:
            label = "Low motion / stopping"
        elif index == reversal:
            label = "Reversal-rich"
        elif index == gripper:
            label = "Gripper event"
        elif index == stopping:
            label = "Stopping / precision"
        elif index == chosen[0]:
            label = "Low motion"
        elif index == chosen[1]:
            label = "Medium motion"
        elif index == chosen[2]:
            label = "High motion"
        else:
            label = "Representative event"
        labels.append((label, index))
    return labels


def figure_a(flow: dict, direct: dict, out: Path) -> None:
    examples = get_examples(flow)
    fig, axes = plt.subplots(len(examples), 6, figsize=(17, 2.25 * len(examples)), sharex=True)
    x = np.arange(32)
    for row, (label, index) in enumerate(examples):
        target = flow["target"][index].numpy()
        prediction = flow["prediction"][index].numpy()
        direct_prediction = direct["prediction"][index].numpy()
        for dim, ax in enumerate(axes[row]):
            setup_axis(ax, JOINTS[dim] if row == 0 else None)
            ax.plot(x, target[:, dim], color=COLORS["Expert"], lw=1.7, label="Expert")
            ax.plot(x, prediction[:, dim], color=COLORS["Full TIPS Flow"], lw=1.5, label="Full TIPS Flow")
            ax.plot(x, direct_prediction[:, dim], color=COLORS["Historical Direct"], lw=1.2, label="Historical Direct")
            if dim == 0:
                ax.set_ylabel(f"{label}\nNormalized action", fontsize=8)
            if row == len(examples) - 1:
                ax.set_xlabel("Horizon step", fontsize=8)
            if row == 0 and dim == 0:
                ax.legend(fontsize=7, loc="best", frameon=False)
    fig.suptitle("Held-out expert and predicted action trajectories (yellow = executed first four)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out / "figure_a_expert_vs_flow.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "figure_a_expert_vs_flow.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_b(flow: dict, direct: dict, out: Path) -> None:
    err = (flow["prediction"][:, :4] - flow["target"][:, :4]).abs().mean((1, 2))
    cases = [(int(err.argmin()), "Lowest held-out First-4 error"), (int(err.argmax()), "Highest held-out First-4 error")]
    fig, axes = plt.subplots(4, 6, figsize=(16, 8), sharex=True)
    x = np.arange(1, 5)
    for slot, (index, label) in enumerate(cases):
        target = flow["target"][index, :4].numpy()
        pred = flow["prediction"][index, :4].numpy()
        direct_pred = direct["prediction"][index, :4].numpy()
        offset = slot * 2
        for dim, ax in enumerate(axes[offset]):
            ax.plot(x, target[:, dim], "o-", color=COLORS["Expert"], lw=1.5, label="Expert")
            ax.plot(x, pred[:, dim], "s-", color=COLORS["Full TIPS Flow"], lw=1.3, label="Full TIPS Flow")
            ax.plot(x, direct_pred[:, dim], "^-", color=COLORS["Historical Direct"], lw=1.1, label="Historical Direct")
            ax.set_title(JOINTS[dim], fontsize=9)
            ax.set_xticks(x)
            ax.grid(axis="y", alpha=0.22)
            if dim == 0:
                ax.set_ylabel(f"{label}\nNormalized action", fontsize=8)
            if slot == 0 and dim == 0:
                ax.legend(fontsize=7, frameon=False)
        for dim, ax in enumerate(axes[offset + 1]):
            flow_err = np.abs(pred[:, dim] - target[:, dim])
            direct_err = np.abs(direct_pred[:, dim] - target[:, dim])
            ax.bar(x - 0.18, flow_err, width=0.34, color=COLORS["Full TIPS Flow"], label="Flow absolute error")
            ax.bar(x + 0.18, direct_err, width=0.34, color=COLORS["Historical Direct"], label="Direct absolute error")
            ax.set_xticks(x)
            ax.grid(axis="y", alpha=0.22)
            if dim == 0:
                ax.set_ylabel("Absolute error", fontsize=8)
            if slot == 0 and dim == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle("First-four action alignment on held-out episodes", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "figure_b_first_four_alignment.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "figure_b_first_four_alignment.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_c(flow: dict, out: Path) -> None:
    snapshots = flow["trajectory_snapshots"]
    available_steps = {int(step) for step in snapshots}
    steps = [step for step in (0, 800, 2000, int(flow["step"])) if step in available_steps]
    first_snapshot = snapshots[str(steps[0])]
    available = {int(index) for index in first_snapshot}
    motion = np.asarray(flow["sample_motion"])
    ranked_available = sorted(available, key=lambda index: motion[index])
    candidates = [
        ranked_available[0],
        ranked_available[len(ranked_available) // 2],
        ranked_available[-1],
    ]
    samples = list(zip(("Low motion", "Medium motion", "High motion"), candidates, strict=True))
    step_colors = plt.cm.viridis(np.linspace(0.18, 0.86, len(steps)))
    fig, axes = plt.subplots(len(samples), 6, figsize=(17, 2.8 * len(samples)), sharex=True)
    x = np.arange(32)
    for row, (label, sample_index) in enumerate(samples):
        target = flow["target"][sample_index].numpy()
        for dim, ax in enumerate(axes[row]):
            setup_axis(ax, JOINTS[dim] if row == 0 else None)
            ax.plot(x, target[:, dim], color=COLORS["Expert"], lw=1.8, label="Expert")
            for color, step in zip(step_colors, steps, strict=True):
                sample = snapshots[str(step)][str(sample_index)]
                y = np.asarray(sample["prediction"])[:, dim]
                ax.plot(x, y, color=color, lw=1.05, alpha=0.9, label=f"Step {step}")
            if dim == 0:
                ax.set_ylabel(f"{label}\nNormalized action", fontsize=8)
            if row == 0 and dim == 0:
                ax.legend(fontsize=6, frameon=False)
            if row == len(samples) - 1:
                ax.set_xlabel("Horizon step", fontsize=8)
    fig.suptitle("Same held-out action chunks at fixed full-student training checkpoints", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "figure_c_training_alignment_progress.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "figure_c_training_alignment_progress.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_d(flow: dict, direct: dict, out: Path, historical: dict | None = None) -> None:
    flow_error = (flow["prediction"] - flow["target"]).abs().mean((0, 2)).numpy()
    direct_error = (direct["prediction"] - direct["target"]).abs().mean((0, 2)).numpy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(1, 33)
    ax.axvspan(0.5, 4.5, color="#f4ce55", alpha=0.28, label="Executed first four")
    ax.plot(x, flow_error, "o-", color=COLORS["Full TIPS Flow"], label="Full TIPS Flow on final validation")
    ax.plot(x, direct_error, "s-", color=COLORS["Historical Direct"], label="Historical Direct on same windows")
    ax.set(xlabel="Horizon step", ylabel="Mean absolute error (normalized action)", title="Held-out action error across the full chunk")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figure_d_horizon_mae.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "figure_d_horizon_mae.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_e(
    flow: dict, direct: dict, out: Path, flow_seed_metrics: list[dict] | None = None
) -> None:
    fmetrics, dmetrics = flow["metrics"], direct["metrics"]
    flow_seed_metrics = flow_seed_metrics or [fmetrics]

    def values_for(path: tuple[str, ...]) -> tuple[list[float], list[float]]:
        values = []
        for metrics in flow_seed_metrics:
            current = metrics
            for key in path:
                current = current[key]
            values.append(float(current))
        mean = np.atleast_1d(np.mean(values, axis=0)).tolist()
        std = np.atleast_1d(
            np.std(values, axis=0, ddof=1 if len(values) > 1 else 0)
        ).tolist()
        return mean, std

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    groups = ["First-4", "Full-32", "High-motion\nFirst-4", "High-motion\nFull-32"]
    fvals = [
        values_for((key,))[0][0]
        for key in ("first4_mae", "full32_mae", "high_motion_first4_mae", "high_motion_full32_mae")
    ]
    ferr = [
        values_for((key,))[1][0]
        for key in ("first4_mae", "full32_mae", "high_motion_first4_mae", "high_motion_full32_mae")
    ]
    dvals = [dmetrics["first4_mae"], dmetrics["full32_mae"], dmetrics["high_motion_first4_mae"], dmetrics["high_motion_full32_mae"]]
    width = 0.36
    xx = np.arange(len(groups))
    axes[0].bar(xx - width / 2, dvals, width, color=COLORS["Historical Direct"], label="Historical Direct")
    axes[0].bar(
        xx + width / 2,
        fvals,
        width,
        yerr=ferr,
        capsize=4,
        color=COLORS["Full TIPS Flow"],
        label="Full TIPS Flow (seed mean ± SD)",
    )
    axes[0].set_xticks(xx, groups, fontsize=8)
    axes[0].set_ylabel("MAE ↓")
    axes[0].set_title("Primary action metrics")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.22)

    names = [name.replace(".pos", "") for name in JOINTS]
    joint_flow = [
        values_for(("per_joint", key, "first4_mae"))[0][0]
        for key in (
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
        )
    ]
    joint_error = [
        values_for(("per_joint", key, "first4_mae"))[1][0]
        for key in (
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
        )
    ]
    joint_direct = [dmetrics["per_joint"][key]["first4_mae"] for key in (
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
    )]
    xj = np.arange(6)
    axes[1].bar(xj - width / 2, joint_direct, width, color=COLORS["Historical Direct"], label="Direct")
    axes[1].bar(xj + width / 2, joint_flow, width, yerr=joint_error, capsize=3, color=COLORS["Full TIPS Flow"], label="Flow mean ± SD")
    axes[1].set_xticks(xj, names, rotation=32, ha="right", fontsize=8)
    axes[1].set_ylabel("First-4 MAE ↓")
    axes[1].set_title("Per-joint First-4 MAE")
    axes[1].grid(axis="y", alpha=0.22)

    tasks = list(fmetrics["per_task"])
    xt = np.arange(len(tasks))
    short = [task.replace("Pick and place the ", "Pick ").replace("Stack red block on the ", "Stack ").replace("Make T character with ", "Make T / ") for task in tasks]
    flow_task = [values_for(("per_task", task, "first4_mae"))[0][0] for task in tasks]
    flow_task_error = [values_for(("per_task", task, "first4_mae"))[1][0] for task in tasks]
    direct_task = [dmetrics["per_task"][task]["first4_mae"] for task in tasks]
    axes[2].bar(xt - width / 2, direct_task, width, color=COLORS["Historical Direct"], label="Direct")
    axes[2].bar(xt + width / 2, flow_task, width, yerr=flow_task_error, capsize=3, color=COLORS["Full TIPS Flow"], label="Flow mean ± SD")
    axes[2].set_xticks(xt, short, rotation=28, ha="right", fontsize=8)
    axes[2].set_ylabel("First-4 MAE ↓")
    axes[2].set_title("Per-task First-4 MAE")
    axes[2].grid(axis="y", alpha=0.22)
    fig.suptitle("Full-student TIPSv2 + Flow held-out action performance (error bars: seed SD)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "figure_e_quantitative_action_performance.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "figure_e_quantitative_action_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_f(flow: dict, direct: dict, out: Path, historical: dict | None = None) -> None:
    error = (flow["prediction"][:, :4] - flow["target"][:, :4]).abs().mean((1, 2))
    examples = [(int(error.argmin()), "Lowest error"), (int(error.argmax()), "Highest error")]
    fig = plt.figure(figsize=(17, 10))
    grid = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.62], hspace=0.32)
    x = np.arange(32)
    for row, (index, label) in enumerate(examples):
        for dim in range(6):
            ax = fig.add_subplot(grid[row, dim])
            setup_axis(ax, JOINTS[dim] if row == 0 else None)
            ax.plot(x, flow["target"][index, :, dim], color=COLORS["Expert"], lw=1.7, label="Expert")
            ax.plot(x, flow["prediction"][index, :, dim], color=COLORS["Full TIPS Flow"], lw=1.4, label="Flow")
            ax.plot(x, direct["prediction"][index, :, dim], color=COLORS["Historical Direct"], lw=1.1, label="Direct")
            if dim == 0:
                ax.set_ylabel(f"{label}\nNormalized action", fontsize=8)
            if row == 0 and dim == 0:
                ax.legend(fontsize=7, frameon=False)
    ax = fig.add_subplot(grid[2, :3])
    fmetrics, dmetrics = flow["metrics"], direct["metrics"]
    labels = ["First-4 MAE", "Full-32 MAE", "High-motion F4", "High-motion F32"]
    fv = [fmetrics["first4_mae"], fmetrics["full32_mae"], fmetrics["high_motion_first4_mae"], fmetrics["high_motion_full32_mae"]]
    dv = [dmetrics["first4_mae"], dmetrics["full32_mae"], dmetrics["high_motion_first4_mae"], dmetrics["high_motion_full32_mae"]]
    positions = np.arange(4)
    ax.bar(positions - 0.18, dv, 0.36, color=COLORS["Historical Direct"], label="Direct")
    ax.bar(positions + 0.18, fv, 0.36, color=COLORS["Full TIPS Flow"], label="Full TIPS Flow")
    ax.set_xticks(positions, labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("MAE ↓")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(grid[2, 3:])
    ax.axis("off")
    variance = fmetrics["temporal_variance_ratio"]
    msg = (
        f"Selected full-student checkpoint: step {flow['step']}\n"
        f"First-4 MAE: {fmetrics['first4_mae']:.4f}  |  Full-32 MAE: {fmetrics['full32_mae']:.4f}\n"
        f"Historical Direct, same windows: {dmetrics['first4_mae']:.4f} / {dmetrics['full32_mae']:.4f}\n"
        f"Temporal variance ratio: {variance:.3f} (expert-relative scale only)\n"
        f"Stopping agreement: {fmetrics['stopping_agreement']:.3f}; reversal F1: {fmetrics['reversal_f1']:.3f}\n"
        "Offline held-out action evidence; no closed-loop robot trial."
    )
    ax.text(0.02, 0.95, msg, va="top", ha="left", fontsize=11, linespacing=1.6)
    fig.suptitle("Full TIPS student: expert alignment summary and representative error extremes", fontsize=15)
    fig.savefig(out / "figure_f_final_action_alignment_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "figure_f_final_action_alignment_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def generate(args: argparse.Namespace) -> None:
    out, train_root = args.output, args.training_dir
    out.mkdir(parents=True, exist_ok=True)
    run, flow, direct = load_results(train_root, args.seed)
    flow_seed_metrics = [flow["metrics"]]
    for seed in args.confirmation_seeds:
        summary_path = train_root / f"seed{seed}" / "summary.json"
        if summary_path.exists():
            flow_seed_metrics.append(json.loads(summary_path.read_text())["metrics"])
    direct_on_same = torch.load(train_root / "historical_direct_on_final_validation.pt", map_location="cpu", weights_only=False)
    direct = {**direct, **direct_on_same}
    figure_a(flow, direct, out)
    figure_b(flow, direct, out)
    figure_c(flow, out)
    figure_d(flow, direct, out)
    figure_e(flow, direct, out, flow_seed_metrics)
    figure_f(flow, direct, out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_DIR / "training")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "figures")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--confirmation-seeds", nargs="*", type=int, default=[20261019])
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
