"""Generate the measured TIPS-Ground-VLA Action-Head report figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/reports/assets/tips_ground_vla_flow_matching"
OUT = BASE / "final"
sys.path.insert(0, "/home/minhhieuano/.codex/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

COLORS = {
    "Direct": "#4b5563",
    "Standard flow": "#d97706",
    "Architecture-aware flow": "#2563eb",
    "Expert": "#111827",
}
PATHS = {
    "Direct": BASE / "broader_seed20261016/direct_seed20261016.pt",
    "Standard flow": BASE / "broader_seed20261016/flow_seed20261016.pt",
    "Architecture-aware flow": BASE / "broader_arch_recovery/architecture_aware_flow_seed20261016.pt",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "legend.frameon": False,
    }
)


def panel(ax, letter: str) -> None:
    ax.annotate(
        letter,
        (0, 1),
        xycoords="axes fraction",
        xytext=(-18, 7),
        textcoords="offset points",
        fontweight="bold",
        fontsize=8,
    )


def save(fig, stem: str, *, labels: bool = True, alignment: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    if alignment:
        require_matplotlib_panel_alignment(
            fig,
            json_out=OUT / f"{stem}.alignment.json",
            overlay_svg=OUT / f"{stem}.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=labels,
            strict=True,
        )
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=400, bbox_inches="tight")
    plt.close(fig)


def load() -> dict:
    return {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in PATHS.items()}


def architecture_box(ax, xy, text, color, width=0.18, height=0.14):
    x, y = xy
    box = mpl.patches.FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.01", fc=color, ec="#374151", lw=0.7
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=6)
    return (x, y, width, height)


def arrow(ax, a, b):
    ax.annotate("", b, a, arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#4b5563"})


def figure_a() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    titles = ["Direct", "Standard conditional flow", "Architecture-aware flow"]
    for index, (ax, title) in enumerate(zip(axes, titles, strict=True)):
        ax.set(xlim=(0, 1), ylim=(0, 1), title=title)
        ax.axis("off")
        architecture_box(ax, (0.04, 0.70), "28 fused\nmultimodal tokens", "#dbeafe", 0.32, 0.16)
        if index == 0:
            architecture_box(ax, (0.46, 0.70), "32 learned\naction queries", "#ede9fe", 0.28, 0.16)
            architecture_box(
                ax, (0.37, 0.38), "6 decoder blocks\nfull cross-attention", "#e0e7ff", 0.42, 0.18
            )
            architecture_box(ax, (0.37, 0.08), "state anchor +\ncumulative increments", "#dcfce7", 0.42, 0.18)
            arrow(ax, (0.20, 0.70), (0.48, 0.55))
            arrow(ax, (0.60, 0.70), (0.60, 0.56))
            arrow(ax, (0.58, 0.38), (0.58, 0.26))
        elif index == 1:
            architecture_box(ax, (0.42, 0.70), "mean pool\n(no source identity)", "#fef3c7", 0.34, 0.16)
            architecture_box(
                ax, (0.33, 0.38), "noisy absolute chunk + t\n4-block vector field", "#ffedd5", 0.52, 0.18
            )
            architecture_box(ax, (0.40, 0.08), "Euler 1–16 NFE\nabsolute actions", "#fee2e2", 0.38, 0.18)
            arrow(ax, (0.36, 0.78), (0.42, 0.78))
            arrow(ax, (0.59, 0.70), (0.59, 0.56))
            arrow(ax, (0.59, 0.38), (0.59, 0.26))
        else:
            architecture_box(ax, (0.40, 0.70), "4 source gates\nA/G/S/dense", "#d1fae5", 0.38, 0.16)
            architecture_box(
                ax,
                (0.29, 0.38),
                "scaled noisy increments + t\n4-block full cross-attention",
                "#dbeafe",
                0.62,
                0.18,
            )
            architecture_box(
                ax, (0.35, 0.08), "state anchor + Euler\ncumulative increments", "#dcfce7", 0.50, 0.18
            )
            arrow(ax, (0.36, 0.78), (0.40, 0.78))
            arrow(ax, (0.59, 0.70), (0.59, 0.56))
            arrow(ax, (0.59, 0.38), (0.59, 0.26))
        panel(ax, chr(97 + index))
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.05, top=0.84, wspace=0.12)
    save(fig, "figure_a_action_head_architectures")


def figure_b(data: dict) -> None:
    target = data["Direct"]["target"]
    motion = (target[:, 1:] - target[:, :-1]).abs().mean((1, 2))
    order = motion.argsort()
    examples = [int(order[3]), int(order[len(order) // 2]), int(order[-4])]
    fig, axes = plt.subplots(6, 3, figsize=(7.2, 8.0), sharex=True, constrained_layout=True)
    for dim in range(6):
        for column, sample in enumerate(examples):
            ax = axes[dim, column]
            ax.axvspan(-0.5, 3.5, color="#fde68a", alpha=0.3)
            ax.plot(target[sample, :, dim], color=COLORS["Expert"], lw=1.3, label="Expert")
            for name, artifact in data.items():
                ax.plot(artifact["prediction"][sample, :, dim], color=COLORS[name], lw=0.8, label=name)
            if dim == 0:
                ax.set_title(("Low", "Medium", "High")[column] + " motion")
            if column == 0:
                ax.set_ylabel(f"Action {dim + 1}")
            if dim == 5:
                ax.set_xlabel("Step (yellow = executed)")
            panel(ax, chr(97 + dim * 3 + column))
    axes[0, 2].legend(fontsize=5, loc="upper right")
    save(fig, "figure_b_expert_predicted_trajectories")


def figure_c(data: dict) -> None:
    names = list(data)
    metrics = [
        "velocity_mae",
        "acceleration_mae",
        "jerk_mae",
        "temporal_variance_ratio",
        "stopping_rate_prediction",
        "reversal_rate_prediction",
    ]
    titles = [
        "Velocity MAE",
        "Acceleration MAE",
        "Jerk MAE",
        "Variance ratio",
        "Stopping rate",
        "Reversal rate",
    ]
    expert = [
        None,
        None,
        None,
        1.0,
        data["Direct"]["result"]["metrics"]["stopping_rate_expert"],
        data["Direct"]["result"]["metrics"]["reversal_rate_expert"],
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3), constrained_layout=True)
    for i, (ax, metric, title) in enumerate(zip(axes.flat, metrics, titles, strict=True)):
        values = [data[name]["result"]["metrics"][metric] for name in names]
        ax.bar(range(3), values, color=[COLORS[name] for name in names])
        if expert[i] is not None:
            ax.axhline(expert[i], color="black", ls="--", lw=0.8, label="Expert")
        ax.set_xticks(
            range(3), ["Direct", "Std flow", "Arch flow"], rotation=20, ha="right", rotation_mode="anchor"
        )
        ax.set_title(title)
        panel(ax, chr(97 + i))
    save(fig, "figure_c_temporal_motion_fidelity")


def continuity(data: dict) -> dict:
    result = {}
    target = data["Direct"]["target"]
    expert_jump = (target[1::2, 0] - target[::2, 3]).abs().mean().item()
    expert_velocity = (
        ((target[1::2, 1] - target[1::2, 0]) - (target[::2, 3] - target[::2, 2])).abs().mean().item()
    )
    for name, artifact in data.items():
        pred = artifact["prediction"]
        result[name] = {
            "boundary_jump": (pred[1::2, 0] - pred[::2, 3]).abs().mean().item(),
            "boundary_jump_error": ((pred[1::2, 0] - pred[::2, 3]) - (target[1::2, 0] - target[::2, 3]))
            .abs()
            .mean()
            .item(),
            "boundary_velocity_jump": ((pred[1::2, 1] - pred[1::2, 0]) - (pred[::2, 3] - pred[::2, 2]))
            .abs()
            .mean()
            .item(),
            "overlap_disagreement": (pred[1::2, :28] - pred[::2, 4:]).abs().mean().item(),
        }
    result["Expert"] = {"boundary_jump": expert_jump, "boundary_velocity_jump": expert_velocity}
    (OUT / "chunk_continuity_metrics.json").write_text(json.dumps(result, indent=2))
    return result


def figure_d(data: dict) -> None:
    values = continuity(data)
    names = list(data)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    for i, (metric, title) in enumerate(
        (
            ("boundary_jump", "Boundary jump"),
            ("boundary_velocity_jump", "Velocity jump"),
            ("overlap_disagreement", "Overlap disagreement"),
        )
    ):
        ax = axes[i]
        ax.bar(range(3), [values[name][metric] for name in names], color=[COLORS[name] for name in names])
        if metric != "overlap_disagreement":
            ax.axhline(values["Expert"][metric], color="black", ls="--", lw=0.8)
        ax.set_xticks(
            range(3), ["Direct", "Std flow", "Arch flow"], rotation=22, ha="right", rotation_mode="anchor"
        )
        ax.set_title(title)
        ax.set_ylabel("Normalized action difference")
        panel(ax, chr(97 + i))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.29, top=0.86, wspace=0.38)
    save(fig, "figure_d_chunk_continuity")


def figure_e() -> None:
    sources = {
        "Direct": [
            BASE / "broader_seed20261016/direct_seed20261016.json",
            BASE / "broader_seed20261017/direct_seed20261017.json",
        ],
        "Standard flow": [BASE / "broader_seed20261016/flow_seed20261016.json"],
        "Architecture-aware flow": [
            BASE / "broader_arch_recovery/architecture_aware_flow_seed20261016.json",
            BASE / "broader_seed20261017/architecture_aware_flow_seed20261017.json",
        ],
    }
    metrics = [
        "first4_mae",
        "full32_mae",
        "high_motion_first4_mae",
        "high_motion_full32_mae",
        "action_delta_mae",
    ]
    titles = ["First-4", "Full-32", "High-motion First-4", "High-motion Full-32", "Action-delta"]
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.25))
    rows = []
    for index, (ax, metric, title) in enumerate(zip(axes, metrics, titles, strict=True)):
        for x, (name, paths) in enumerate(sources.items()):
            vals = [json.loads(path.read_text())["metrics"][metric] for path in paths]
            ax.bar(x, np.mean(vals), color=COLORS[name], alpha=0.75)
            ax.scatter([x] * len(vals), vals, color="black", s=8, zorder=3)
            rows.extend(
                {"head": name, "metric": metric, "value": value, "source": path.name}
                for value, path in zip(vals, paths, strict=True)
            )
        ax.set_xticks(range(3), ["Direct", "Std", "Arch"], rotation=25, ha="right", rotation_mode="anchor")
        ax.set_title(title)
        ax.set_ylabel("MAE")
        panel(ax, chr(97 + index))
    (OUT / "accuracy_source_data.json").write_text(json.dumps(rows, indent=2))
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.28, top=0.84, wspace=0.38)
    save(fig, "figure_e_action_accuracy")


def figure_f() -> None:
    rows = json.loads((BASE / "sampling_architecture_aware.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for noise, style in (("random", "-o"), ("zero", "--s")):
        selected = [row for row in rows if row["inference_noise"] == noise]
        steps = [row["flow_steps"] for row in selected]
        axes[0].plot(steps, [row["first4_mae"] for row in selected], style, label=noise)
        axes[1].plot(steps, [row["full32_mae"] for row in selected], style, label=noise)
        axes[2].plot(steps, [row["full_student_latency_ms"] for row in selected], style, label=noise)
    for i, (ax, title, ylabel) in enumerate(
        zip(
            axes,
            ("Immediate accuracy", "Full horizon", "End-to-end latency"),
            ("First-4 MAE", "Full-32 MAE", "Milliseconds"),
            strict=True,
        )
    ):
        ax.set(xlabel="Euler NFEs", ylabel=ylabel, title=title, xscale="log")
        ax.set_xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
        panel(ax, chr(97 + i))
    axes[0].legend(title="Initial noise", fontsize=6)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.85, wspace=0.36)
    save(fig, "figure_f_sampling_study")


def figure_g() -> None:
    paths = {
        "Direct": BASE / "broader_seed20261016/direct_seed20261016.json",
        "Standard flow": BASE / "broader_seed20261016/flow_seed20261016.json",
        "Architecture-aware flow": BASE / "broader_arch_recovery/architecture_aware_flow_seed20261016.json",
    }
    loaded = {name: json.loads(path.read_text()) for name, path in paths.items()}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.3), constrained_layout=True)
    specs = (
        ("action_loss", "Training endpoint loss"),
        ("first4_mae", "Held-out First-4"),
        ("full32_mae", "Held-out Full-32"),
        ("grad_norm", "Pre-clip gradient norm"),
    )
    for i, (ax, (metric, title)) in enumerate(zip(axes.flat, specs, strict=True)):
        for name, payload in loaded.items():
            curve = payload["curve"]
            ax.plot(
                [row["step"] for row in curve], [row[metric] for row in curve], color=COLORS[name], label=name
            )
        ax.set(xlabel="Update", ylabel=metric.replace("_", " "), title=title)
        panel(ax, chr(97 + i))
    axes[0, 0].legend(fontsize=6)
    save(fig, "figure_g_training_validation_curves")


def figure_h() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boxes = [
        ((0.02, 0.72), "Top RGB", "#dbeafe", 0.12),
        ((0.02, 0.52), "Wrist RGB", "#dbeafe", 0.12),
        ((0.02, 0.30), "Instruction", "#fef3c7", 0.12),
        ((0.02, 0.10), "Robot state", "#dcfce7", 0.12),
        ((0.19, 0.62), "ViT-S/16", "#dbeafe", 0.14),
        ((0.19, 0.30), "MiniLM-L12-H384", "#fef3c7", 0.17),
        ((0.19, 0.08), "4 state tokens", "#dcfce7", 0.17),
        ((0.40, 0.66), "8-query Alignment", "#ede9fe", 0.16),
        ((0.40, 0.46), "8-slot Grounding", "#ede9fe", 0.16),
        ((0.40, 0.24), "Dense visual bypass", "#dbeafe", 0.16),
        ((0.61, 0.44), "Multimodal fusion", "#e0e7ff", 0.16),
        ((0.81, 0.44), "Direct decoder\nstate-anchored deltas", "#fee2e2", 0.17),
        ((0.81, 0.17), "32 actions → execute 4\nobserve + replan", "#dcfce7", 0.17),
        (
            (0.39, 0.02),
            "Optional frozen TIPS/DINO teachers\ntraining only; objectives disabled here",
            "#f3f4f6",
            0.30,
        ),
    ]
    for xy, text, color, width in boxes:
        architecture_box(ax, xy, text, color, width, 0.12)
    for a, b in [
        ((0.14, 0.78), (0.19, 0.70)),
        ((0.14, 0.58), (0.19, 0.68)),
        ((0.14, 0.36), (0.19, 0.36)),
        ((0.14, 0.16), (0.19, 0.14)),
        ((0.33, 0.68), (0.40, 0.72)),
        ((0.36, 0.36), (0.40, 0.52)),
        ((0.36, 0.14), (0.61, 0.50)),
        ((0.56, 0.72), (0.61, 0.54)),
        ((0.56, 0.52), (0.61, 0.50)),
        ((0.56, 0.30), (0.61, 0.48)),
        ((0.77, 0.50), (0.81, 0.50)),
        ((0.90, 0.44), (0.90, 0.29)),
    ]:
        arrow(ax, a, b)
    ax.annotate(
        "replan",
        (0.81, 0.22),
        (0.70, 0.14),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "connectionstyle": "arc3,rad=-0.35"},
        fontsize=6,
    )
    ax.set_title("Selected deployable TIPS-Ground-VLA (Flow Matching rejected)", fontweight="bold")
    save(fig, "figure_h_final_architecture", labels=False, alignment=False)


def main() -> None:
    data = load()
    figure_a()
    figure_b(data)
    figure_c(data)
    figure_d(data)
    figure_e()
    figure_f()
    figure_g()
    figure_h()


if __name__ == "__main__":
    main()
