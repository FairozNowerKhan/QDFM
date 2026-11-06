import argparse, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

METHODS = ["cql", "iql", "awac", "bcq", "brac", "greedyq", "qdfm"]
DISPLAY = {"cql": "CQL", "iql": "IQL", "awac": "AWAC", "bcq": "BCQ",
           "brac": "BRAC", "greedyq": "GreedyQ", "qdfm": "QDFM (Ours)"}
COLORS = {"cql": "#888888", "iql": "#6699bb", "awac": "#5577aa", "bcq": "#66aa77",
          "brac": "#aa8855", "greedyq": "#9977aa", "qdfm": "#cc3333"}
MARKERS = {"cql": "s", "iql": "^", "awac": "v", "bcq": "D",
           "brac": "p", "greedyq": "h", "qdfm": "o"}
K_VALUES = [2, 3, 4, 5]
HEATMAP_METHODS = ["cql", "awac", "bcq", "qdfm"]
_GOAL_X = {2: [2,8], 3: [1,5,9], 4: [0,3,7,10], 5: [0,2,5,8,10]}


def load_all(results_dir):
    data = defaultdict(lambda: defaultdict(list))
    for f in sorted(results_dir.glob("metrics_K*_*.json")):
        with open(f) as fh:
            m = json.load(fh)
        data[m["method"]][m["K"]].append(m)
    return data


def load_visits(results_dir, method, K, seed):
    p = results_dir / f"visits_K{K}_{method}_s{seed}.npy"
    if p.exists():
        return np.load(p)
    return None


def agg(data, method, K, key):
    vals = [d[key] for d in data[method][K]]
    if not vals:
        return (np.nan, np.nan)
    return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def _draw_heatmap(ax, visits, K, title, cmap="YlOrRd"):
    v = visits.astype(np.float64)
    vmax = max(v.max(), 1)
    ax.imshow(v, origin="lower", cmap=cmap, aspect="equal",
              vmin=0, vmax=vmax, interpolation="nearest", rasterized=True)
    trap_rect = Rectangle((4.5, 2.5), 1, 7, linewidth=1.2,
                           edgecolor="black", facecolor="#cccccc", zorder=2)
    ax.add_patch(trap_rect)
    for gx in _GOAL_X[K]:
        ax.plot(gx, 10, marker="*", color="gold", markersize=12,
                markeredgecolor="black", markeredgewidth=0.6, zorder=3)
    ax.plot(5, 0, marker="o", color="lime", markersize=8,
            markeredgecolor="black", markeredgewidth=0.6, zorder=3)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10])
    ax.set_title(title, pad=6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _line_plot(ax, data, key, ylabel, ylim=None):
    for m in METHODS:
        means, stds = [], []
        for K in K_VALUES:
            mu, sd = agg(data, m, K, key)
            means.append(mu)
            stds.append(sd)
        means = np.array(means)
        stds = np.array(stds)
        lw = 2.8 if m == "qdfm" else 1.5
        ax.errorbar(K_VALUES, means, yerr=stds, label=DISPLAY[m],
                    color=COLORS[m], marker=MARKERS[m], linewidth=lw,
                    capsize=3, linestyle="-", zorder=5 if m == "qdfm" else 2)
    ax.set_xlabel("Number of goals (K)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(K_VALUES)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="best", framealpha=1.0, ncol=2)
    ax.grid(True, color="#dddddd", linewidth=0.5)


def fig_heatmaps(results_dir, fig_dir, K=4, seed=42):
    data = load_all(results_dir)
    fig, axes = plt.subplots(1, len(HEATMAP_METHODS),
                             figsize=(3.4 * len(HEATMAP_METHODS), 3.6))
    for i, m in enumerate(HEATMAP_METHODS):
        v = load_visits(results_dir, m, K, seed)
        if v is None:
            v = np.zeros((11, 11), dtype=np.int32)
        mc = agg(data, m, K, "mode_coverage")
        title = f"{DISPLAY[m]}\nmode cov = {mc[0]:.2f}"
        _draw_heatmap(axes[i], v, K, title)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(fig_dir / "fig_heatmaps.eps", format="eps")
    plt.close(fig)


def fig_mode_coverage(data, fig_dir):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    _line_plot(ax, data, "mode_coverage", "Mode Coverage", ylim=(-0.05, 1.12))
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_mode_coverage.eps", format="eps")
    plt.close(fig)


def fig_trap_rate(data, fig_dir):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    _line_plot(ax, data, "trap_rate", "Trap Entry Rate", ylim=(-0.02, None))
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_trap_rate.eps", format="eps")
    plt.close(fig)


def fig_return(data, fig_dir):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    _line_plot(ax, data, "mean_return", "Mean Return")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_return.eps", format="eps")
    plt.close(fig)


def fig_main(results_dir, data, fig_dir, K_heat=4, seed_heat=42):
    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(2, len(HEATMAP_METHODS),
                          height_ratios=[1.0, 0.9], hspace=0.42, wspace=0.30)
    for i, m in enumerate(HEATMAP_METHODS):
        ax = fig.add_subplot(gs[0, i])
        v = load_visits(results_dir, m, K_heat, seed_heat)
        if v is None:
            v = np.zeros((11, 11), dtype=np.int32)
        mc = agg(data, m, K_heat, "mode_coverage")
        panel = chr(ord("a") + i)
        title = f"({panel}) {DISPLAY[m]}\nmode cov = {mc[0]:.2f}"
        _draw_heatmap(ax, v, K_heat, title)
    ax_bot = fig.add_subplot(gs[1, :])
    _line_plot(ax_bot, data, "mode_coverage", "Mode Coverage", ylim=(-0.05, 1.12))
    ax_bot.set_title("(e) Mode coverage vs. number of goals", pad=8)
    fig.savefig(fig_dir / "fig_main.eps", format="eps")
    plt.close(fig)


def fig_appendix(results_dir, data, fig_dir, K_heat=4, seed_heat=42):
    fig = plt.figure(figsize=(14, 10.5))
    outer = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)
    inner_a = outer[0, 0].subgridspec(1, len(HEATMAP_METHODS), wspace=0.35)
    for i, m in enumerate(HEATMAP_METHODS):
        ax = fig.add_subplot(inner_a[0, i])
        v = load_visits(results_dir, m, K_heat, seed_heat)
        if v is None:
            v = np.zeros((11, 11), dtype=np.int32)
        mc = agg(data, m, K_heat, "mode_coverage")
        _draw_heatmap(ax, v, K_heat, DISPLAY[m])
        if i > 0:
            ax.set_ylabel("")
    ax_b = fig.add_subplot(outer[0, 1])
    _line_plot(ax_b, data, "mode_coverage", "Mode Coverage", ylim=(-0.05, 1.12))
    ax_b.set_title("(b) Mode coverage vs K", pad=6)
    ax_c = fig.add_subplot(outer[1, 0])
    _line_plot(ax_c, data, "trap_rate", "Trap Entry Rate", ylim=(-0.02, None))
    ax_c.set_title("(c) Trap entry rate vs K", pad=6)
    ax_d = fig.add_subplot(outer[1, 1])
    _line_plot(ax_d, data, "mean_return", "Mean Return")
    ax_d.set_title("(d) Mean return vs K", pad=6)
    fig.savefig(fig_dir / "fig_appendix.eps", format="eps")
    plt.close(fig)


def latex_table(data, fig_dir):
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Multi-goal gridworld results. Mean $\pm$ std over 5 seeds, "
                 r"200 eval episodes each.}")
    lines.append(r"\label{tab:multigoal}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lcccccccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{$K=2$} & \multicolumn{2}{c}{$K=3$}"
                 r" & \multicolumn{2}{c}{$K=4$} & \multicolumn{2}{c}{$K=5$} \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    lines.append(r"Method & Return & Mode Cov & Return & Mode Cov"
                 r" & Return & Mode Cov & Return & Mode Cov \\")
    lines.append(r"\midrule")
    for m in METHODS:
        if m == "qdfm":
            row = r"QDFM \textbf{(Ours)}"
        else:
            row = DISPLAY[m]
        for K in K_VALUES:
            ret_m, ret_s = agg(data, m, K, "mean_return")
            mc_m, mc_s = agg(data, m, K, "mode_coverage")
            if np.isnan(ret_m):
                row += r" & N/A & N/A"
            else:
                row += f" & {ret_m:.2f}" + r"{\scriptsize$\pm$" + f"{ret_s:.1f}" + "}"
                row += f" & {mc_m:.2f}"
        row += r" \\"
        lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    (fig_dir / "table_multigoal.tex").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="results_multigoal")
    ap.add_argument("--fig_dir", type=str, default="figures")
    ap.add_argument("--K_heat", type=int, default=4)
    ap.add_argument("--seed_heat", type=int, default=42)
    args = ap.parse_args()

    rd = Path(args.dir)
    fd = Path(args.fig_dir)
    fd.mkdir(parents=True, exist_ok=True)

    data = load_all(rd)
    if not data:
        sys.exit(1)

    fig_heatmaps(rd, fd, K=args.K_heat, seed=args.seed_heat)
    fig_mode_coverage(data, fd)
    fig_trap_rate(data, fd)
    fig_return(data, fd)
    fig_main(rd, data, fd, K_heat=args.K_heat, seed_heat=args.seed_heat)
    fig_appendix(rd, data, fd, K_heat=args.K_heat, seed_heat=args.seed_heat)
    latex_table(data, fd)


if __name__ == "__main__":
    main()
