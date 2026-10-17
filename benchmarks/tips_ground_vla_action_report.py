"""Build the final evidence bundle from saved TIPS-Ground-VLA experiments."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/reports/assets/tips_ground_vla_action_improvement/final"
TEXT = ROOT / "docs/reports/assets/tips_ground_vla_text_transfer"
IMPROVE = ROOT / "docs/reports/assets/tips_ground_vla_action_improvement"
TEACHER = ROOT / "docs/reports/assets/tips_ground_vla_teacher_integration"
sys.path.insert(0, "/home/minhhieuano/.codex/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

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
COLORS = {"Action-only": "#6b7280", "TIPS semantic": "#e69f00", "TIPS text": "#cc79a7", "Semantic decay": "#0072b2"}


def load_json(path: Path):
    return json.loads(path.read_text())


def save(fig, stem: str, *, labels: bool = True):
    fig.canvas.draw()
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
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter: str):
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


def aggregate_confirmation():
    rows = []
    names = {"action": "Action-only", "semantic": "TIPS semantic", "text": "TIPS text"}
    for variant, display in names.items():
        for seed in (20261016, 20261017):
            data = load_json(TEXT / f"confirmation/{variant}_seed{seed}.json")
            row = {"variant": display, "seed": seed, **{k: v for k, v in data["metrics"].items() if k != "per_task"}}
            row["per_task"] = data["metrics"]["per_task"]
            rows.append(row)
    decay = load_json(IMPROVE / "confirmation/semantic_decay_seed20261017.json")
    rows.append({"variant": "Semantic decay", "seed": 20261017, **{k: v for k, v in decay["metrics"].items() if k != "per_task"}, "per_task": decay["metrics"]["per_task"]})
    return rows


def figure_a(rows):
    metrics = ["first4_mae", "full32_mae", "high_motion_first4_mae", "high_motion_full32_mae"]
    titles = ["First-4 MAE", "Full-32 MAE", "High-motion First-4", "High-motion Full-32"]
    order = list(COLORS)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5), constrained_layout=True)
    for letter, ax, metric, title in zip("abcd", axes.flat, metrics, titles, strict=True):
        means, errors = [], []
        for name in order:
            values = [r[metric] for r in rows if r["variant"] == name]
            means.append(np.mean(values))
            errors.append(np.std(values, ddof=1) if len(values) > 1 else 0)
        ax.bar(range(len(order)), means, yerr=errors, color=[COLORS[x] for x in order], capsize=3)
        ax.set_xticks(range(len(order)), [x.replace(" ", "\n") for x in order])
        ax.set_ylabel("Normalized MAE (lower is better)")
        ax.set_title(title)
        panel(ax, letter)
    save(fig, "figure_a_overall_action_performance")

    tasks = list(rows[0]["per_task"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for letter, ax, task in zip("abc", axes, tasks, strict=True):
        for i, name in enumerate(order):
            values = [r["per_task"][task]["full32_mae"] for r in rows if r["variant"] == name]
            ax.bar(i, np.mean(values), yerr=np.std(values, ddof=1) if len(values) > 1 else 0, color=COLORS[name], capsize=2)
        ax.set_xticks(range(len(order)), ["Action", "Semantic", "Text", "Decay"], rotation=25, ha="right", rotation_mode="anchor")
        ax.set_title(task.replace("Pick and place the red block into the pen holder", "Pick/place red block").replace("Stack red block on the blue block", "Stack red on blue").replace("Make T character with green block", "Make T with green"))
        ax.set_ylabel("Full-32 MAE")
        panel(ax, letter)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.28, top=0.82, wspace=0.35)
    save(fig, "figure_a_per_task")


def figure_b():
    paths = {
        "Action-only": TEXT / "confirmation/action_seed20261017.pt",
        "TIPS semantic": TEXT / "confirmation/semantic_seed20261017.pt",
        "Semantic decay": IMPROVE / "confirmation/semantic_decay_seed20261017.pt",
    }
    loaded = {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in paths.items()}
    target = loaded["Action-only"]["target"]
    motion = (target[:, 1:] - target[:, :-1]).abs().mean((1, 2))
    examples = [int(motion.argmin()), int(motion.argmax())]
    fig, axes = plt.subplots(6, 2, figsize=(7.2, 8.2), sharex=True, constrained_layout=True)
    for dim in range(6):
        for col, index in enumerate(examples):
            ax = axes[dim, col]
            ax.axvspan(-0.5, 3.5, color="#f0e442", alpha=0.16)
            ax.plot(target[index, :, dim], color="black", lw=1.5, label="Expert")
            for name, artifact in loaded.items():
                ax.plot(artifact["prediction"][index, :, dim], color=COLORS[name], lw=1, label=name)
            ax.set_ylabel(f"Action {dim + 1}")
            if dim == 0:
                ax.set_title("Low-motion held-out chunk" if col == 0 else "High-motion held-out chunk")
            if dim == 5:
                ax.set_xlabel("Prediction step (yellow = executed First-4)")
            panel(ax, chr(ord("a") + dim * 2 + col))
    axes[0, 1].legend(ncol=2, fontsize=6, loc="upper right")
    save(fig, "figure_b_expert_predicted_trajectories")


def figure_d():
    correct = load_json(TEXT / "pilot/text_seed20261017.json")
    shuffled = load_json(TEXT / "pilot/text_shuffled_seed20261017.json")
    health = load_json(TEACHER / "transfer_health.json")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7), constrained_layout=True)
    ax = axes[0, 0]
    for data, label, color in ((correct, "Correct TIPS text", "#0072b2"), (shuffled, "Shuffled TIPS text", "#d55e00")):
        ax.plot([x["step"] for x in data["curve"]], [x["teacher_student_cosine"] for x in data["curve"]], marker="o", color=color, label=label)
    ax.set(xlabel="Update", ylabel="Teacher–student cosine", title="Text representation transfer")
    ax.legend()
    panel(ax, "a")
    ax = axes[0, 1]
    x = np.arange(2)
    ax.bar(x - 0.18, [correct["metrics"]["first4_mae"], correct["metrics"]["full32_mae"]], 0.36, label="Correct", color="#0072b2")
    ax.bar(x + 0.18, [shuffled["metrics"]["first4_mae"], shuffled["metrics"]["full32_mae"]], 0.36, label="Shuffled", color="#d55e00")
    ax.set_xticks(x, ["First-4", "Full-32"])
    ax.set_ylabel("MAE")
    ax.set_title("Correct vs shuffled control")
    ax.legend()
    panel(ax, "b")
    ax = axes[1, 0]
    labels = ["Correct", "Wrong", "Uniform"]
    values = [health["tips"]["fixed_validation"][f"tips_{x}"] for x in ("correct", "wrong", "uniform")]
    ax.bar(labels, values, color=["#0072b2", "#d55e00", "#999999"])
    ax.set_ylabel("Semantic KL (lower is better)")
    ax.set_title("Held-out instruction discrimination")
    panel(ax, "c")
    ax = axes[1, 1]
    names = ["Action-only", "TIPS semantic", "TIPS + DINO"]
    values = [health[x]["fixed_validation"]["instruction_action_delta"] for x in ("action", "tips", "joint")]
    ax.bar(names, values, color=[COLORS["Action-only"], COLORS["TIPS semantic"], "#56b4e9"])
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Mean |action(correct) − action(wrong)|")
    ax.set_title("Instruction sensitivity (not correctness)")
    panel(ax, "d")
    save(fig, "figure_d_language_representation_sensitivity")


def figure_e():
    datasets = {
        "Action-only": load_json(TEXT / "confirmation/action_seed20261017.json"),
        "TIPS semantic": load_json(TEXT / "confirmation/semantic_seed20261017.json"),
        "TIPS text": load_json(TEXT / "confirmation/text_seed20261017.json"),
        "Semantic decay": load_json(IMPROVE / "confirmation/semantic_decay_seed20261017.json"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7), constrained_layout=True)
    specs = (("full32_mae", "Held-out Full-32 MAE"), ("first4_mae", "Held-out First-4 MAE"), ("action_loss", "Training action loss"), ("grad_norm", "Pre-clip gradient norm"))
    for letter, ax, (metric, title) in zip("abcd", axes.flat, specs, strict=True):
        for name, data in datasets.items():
            ax.plot([x["step"] for x in data["curve"]], [x[metric] for x in data["curve"]], color=COLORS[name], lw=1.2, label=name)
        ax.set(xlabel="Update", ylabel=metric.replace("_", " "), title=title)
        panel(ax, letter)
    axes[0, 0].legend(fontsize=6, ncol=2)
    save(fig, "figure_e_training_validation_curves")


def figure_f():
    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    ax.set_axis_off()
    boxes = [
        (0.02, 0.63, 0.14, 0.22, "Two RGB\ncameras", "#dbeafe"),
        (0.02, 0.35, 0.14, 0.18, "Robot state", "#dcfce7"),
        (0.02, 0.09, 0.14, 0.18, "Instruction", "#fef3c7"),
        (0.22, 0.58, 0.16, 0.28, "ViT-S/16\n+ dense bypass", "#dbeafe"),
        (0.22, 0.28, 0.16, 0.18, "4 state tokens", "#dcfce7"),
        (0.22, 0.04, 0.16, 0.18, "MiniLM-L12-H384", "#fef3c7"),
        (0.44, 0.52, 0.15, 0.24, "8-query Alignment\n+ 8-slot Grounding", "#ede9fe"),
        (0.64, 0.45, 0.14, 0.28, "Multimodal fusion\n+ dense visual path", "#e0e7ff"),
        (0.83, 0.45, 0.15, 0.28, "32-step decoder\nexecute 4 + replan", "#fee2e2"),
        (0.43, 0.08, 0.17, 0.18, "Frozen TIPSv2\ntraining only", "#f3f4f6"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(mpl.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=color, ec="#374151", lw=0.8, transform=ax.transAxes))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", transform=ax.transAxes)
    arrows = [((0.16, 0.74), (0.22, 0.72)), ((0.16, 0.44), (0.22, 0.37)), ((0.16, 0.18), (0.22, 0.13)), ((0.38, 0.70), (0.44, 0.64)), ((0.38, 0.37), (0.64, 0.54)), ((0.38, 0.13), (0.44, 0.58)), ((0.59, 0.64), (0.64, 0.59)), ((0.78, 0.59), (0.83, 0.59))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 1, "color": "#374151"})
    ax.annotate("semantic/text targets", xy=(0.50, 0.52), xytext=(0.51, 0.26), xycoords=ax.transAxes, ha="center", fontsize=6, color="#d55e00", arrowprops={"arrowstyle": "->", "ls": "--", "color": "#d55e00"})
    ax.text(0.5, 0.96, "Final retained system: teachers supervise training only; deployed student is teacher-free", ha="center", va="top", transform=ax.transAxes, fontweight="bold")
    save(fig, "figure_f_final_architecture", labels=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = aggregate_confirmation()
    flat = [{k: v for k, v in row.items() if k != "per_task"} for row in rows]
    with (OUT / "matched_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (OUT / "matched_metrics.json").write_text(json.dumps(rows, indent=2))
    figure_a(rows)
    figure_b()
    figure_d()
    figure_e()
    figure_f()
    shutil.copy2(TEACHER / "figure_a_tips_semantic_transfer.png", OUT / "figure_c_tips_semantic_grounding.png")


if __name__ == "__main__":
    main()
