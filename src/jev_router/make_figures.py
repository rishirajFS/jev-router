"""Renders the report figure from the saved result files (no API calls)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jev_router.frontier import lower_hull  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SURFACE, INK, INK_2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#8a8983"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated all-pairs (slots 1-3)


def _per_1k(cost: float, n: int) -> float:
    return 1000 * cost / n


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fresh_figure() -> Path:
    """Accuracy-vs-cost and frontier gaps on the untouched fresh set (one frozen evaluation)."""
    fresh = json.loads((ROOT / "data/results/fresh_heldout.json").read_text())
    grp = fresh["groups"]["combined"]
    n = grp["n"]
    models = ("claude-haiku-4-5", "claude-sonnet-5-5", "claude-opus-5-5")
    names = {"claude-haiku-4-5": "Haiku only", "claude-sonnet-5-5": "Sonnet only", "claude-opus-5-5": "Opus only"}
    points = {m: (_per_1k(grp["baselines"][m]["total_cost"], n), 100 * grp["baselines"][m]["accuracy"]) for m in models}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.15, 1]})
    _style(ax1)
    hull = lower_hull([(a, c) for c, a in points.values()])
    ax1.plot([c for a, c in hull], [a for a, c in hull], "--", color=NEUTRAL, lw=1.6, zorder=1, label="best mix of single models")
    offsets = {models[0]: (8, -12), models[1]: (-50, 8), models[2]: (-14, 10)}
    for m, (c, a) in points.items():
        ax1.scatter(c, a, s=70, color=NEUTRAL, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax1.annotate(names[m], (c, a), textcoords="offset points", xytext=offsets[m], fontsize=9, color=INK_2)
    learned = [r for r in grp["learned"] if r["rule"] == "threshold"]
    for rows, color, label in (
        ([(_per_1k(r["result"]["total_cost"], n), 100 * r["result"]["accuracy"]) for r in grp["old_threshold"]], ORANGE, "Threshold router (difficulty only)"),
        ([(_per_1k(r["result"]["total_cost"], n), 100 * r["result"]["accuracy"]) for r in learned], BLUE, "Learned router (frozen)"),
    ):
        ax1.plot([c for c, a in rows], [a for c, a in rows], "-", color=color, lw=2, zorder=2, label=label)
        ax1.scatter([c for c, a in rows], [a for c, a in rows], s=60, color=color, zorder=4, edgecolor=SURFACE, linewidth=1.5)
    ax1.set_xlabel("cost per 1,000 requests (USD, incl. Jev)", color=INK_2, fontsize=10)
    ax1.set_ylabel("accuracy (%)", color=INK_2, fontsize=10)
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    ax1.set_title("Fresh set never used for design (n=%d, evaluated once)" % n, loc="left", fontsize=11, color=INK, pad=10)

    _style(ax2)
    ax2.axhline(0, color=INK_2, lw=1)
    floors = (0.99, 0.97, 0.95)
    series = (
        ("Threshold router", ORANGE, {r["quality_floor"]: r["frontier_gap_pct"] for r in grp["old_threshold"]}),
        ("Learned, threshold rule", BLUE, {r["quality_floor"]: r["frontier_gap_pct"] for r in grp["learned"] if r["rule"] == "threshold"}),
        ("Learned, expected-loss rule", AQUA, {r["quality_floor"]: r["frontier_gap_pct"] for r in grp["learned"] if r["rule"] == "loss"}),
    )
    width = 0.26
    for k, (label, color, gaps) in enumerate(series):
        xs = [i + (k - 1) * (width + 0.02) for i in range(3)]
        vals = [gaps[f] for f in floors]
        ax2.bar(xs, vals, width=width, color=color, label=label, zorder=3, linewidth=0)
        for x, v in zip(xs, vals):
            ax2.text(x, v - 1.0 if v < 0 else v + 1.0, f"{v:+.0f}%", ha="center", va="top" if v < 0 else "bottom", fontsize=8.5, color=INK)
    ax2.set_xticks(range(3), [f"keep {int(100 * f)}% of\nOpus quality" for f in floors])
    ax2.set_ylabel("cost vs best single-model mix (%)\nbelow zero = cheaper", color=INK_2, fontsize=10)
    ax2.set_title("Cost gap to the best single-model mix (fresh set)", loc="left", fontsize=11, color=INK, pad=10)
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    ax2.set_ylim(-26, 14)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.012, "One frozen evaluation on 594 items never scored or used for design. Router refit exactly as before on the original dev half. "
             "Gap is in percent of the frontier's cost at the same accuracy.", fontsize=8.5, color=INK_2)
    out = ROOT / "docs/figures/fresh.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    return out


def main() -> int:
    print(fresh_figure())
    v2 = json.loads((ROOT / "data/results/router_v2.json").read_text())
    multi = json.loads((ROOT / "data/results/router_v2_multi_seed.json").read_text())
    n = v2["n_test"]
    base = v2["baselines"]
    primary = v2["primary_feature_set"]
    learned = [r for r in v2["learned"][primary]["rows"] if r["rule"] == "threshold"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.15, 1]})
    _style(ax1)
    points = {k: (_per_1k(v["total_cost"], n), 100 * v["accuracy"]) for k, v in base.items()}
    hull = lower_hull([(a, c) for c, a in points.values()])
    ax1.plot([c for a, c in hull], [a for a, c in hull], "--", color=NEUTRAL, lw=1.6, zorder=1)
    offsets = {"always_haiku": (8, -12), "always_sonnet": (-46, 8), "always_opus": (-14, 10)}
    for name, (c, a) in points.items():
        ax1.scatter(c, a, s=70, color=NEUTRAL, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax1.annotate(name.replace("always_", "").title() + " only", (c, a), textcoords="offset points", xytext=offsets[name], fontsize=9, color=INK_2)
    for rows, color, label in (
        ([{"c": _per_1k(r["jev"]["total_cost"], n), "a": 100 * r["jev"]["accuracy"]} for r in v2["old_threshold"]], ORANGE, "Threshold router (difficulty only)"),
        ([{"c": _per_1k(r["result"]["total_cost"], n), "a": 100 * r["result"]["accuracy"]} for r in learned], BLUE, "Learned router (6 Jev questions)"),
    ):
        xs, ys = [r["c"] for r in rows], [r["a"] for r in rows]
        ax1.plot(xs, ys, "-", color=color, lw=2, zorder=2, label=label)
        ax1.scatter(xs, ys, s=60, color=color, zorder=4, edgecolor=SURFACE, linewidth=1.5)
    ax1.set_xlabel("cost per 1,000 requests (USD, incl. Jev)", color=INK_2, fontsize=10)
    ax1.set_ylabel("accuracy (%)", color=INK_2, fontsize=10)
    ax1.plot([], [], "--", color=NEUTRAL, lw=1.6, label="best mix of single models")
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    ax1.set_title("Accuracy vs cost (held-out half of one split, n=%d)" % n, loc="left", fontsize=11, color=INK, pad=10)

    _style(ax2)
    ax2.axhline(0, color=INK_2, lw=1)
    floors = ("0.99", "0.97", "0.95")
    series = (("old_threshold:%s", "Threshold router", ORANGE), ("primary:threshold:%s", "Learned, threshold rule", BLUE),
              ("primary:loss:%s", "Learned, expected-loss rule", AQUA))
    width = 0.26
    for k, (pattern, label, color) in enumerate(series):
        xs = [i + (k - 1) * (width + 0.02) for i in range(3)]
        vals = [multi[pattern % f]["gap_mean"] for f in floors]
        errs = [multi[pattern % f]["gap_std"] for f in floors]
        ax2.bar(xs, vals, width=width, color=color, label=label, zorder=3, linewidth=0)
        ax2.errorbar(xs, vals, yerr=errs, fmt="none", ecolor=INK_2, elinewidth=1, capsize=3, zorder=4)
        for x, v, e, f in zip(xs, vals, errs, floors):
            share = 100 * multi[pattern % f]["share_cheaper_than_frontier"]
            y = v - e - 1.2 if v < 0 else v + e + 1.2
            ax2.text(x, y, f"{v:+.0f}%\n({share:.0f}%)", ha="center", va="top" if v < 0 else "bottom", fontsize=8, color=INK)
    ax2.set_xticks(range(3), [f"keep {int(100 * float(f))}% of\nOpus quality" for f in floors])
    ax2.set_ylabel("cost vs best single-model mix (%)\nbelow zero = cheaper", color=INK_2, fontsize=10)
    ax2.set_title("Cost gap to the best single-model mix (20 random splits)", loc="left", fontsize=11, color=INK, pad=10)
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    ax2.set_ylim(-46, 40)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.012, "Right: bars are the mean over 20 splits, whiskers 1 std, (%) is the share of splits where the router beats the best mix. "
             "Left: points up-and-left of the dashed line beat any mix of single models.", fontsize=8.5, color=INK_2)
    out = ROOT / "docs/figures/results.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
