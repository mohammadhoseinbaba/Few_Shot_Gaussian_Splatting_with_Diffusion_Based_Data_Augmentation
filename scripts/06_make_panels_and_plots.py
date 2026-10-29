"""Metric plots, results table and side-by-side comparison panels from the GS runs.

Reads  <root>/gs_runs/results.jsonl  and  <root>/gs_runs/<condition>/test/ours_*/{gt,renders}
Writes <out>/plot_{PSNR,SSIM,LPIPS}_few<N>.png, <out>/cost_few<N>.png, <out>/results_table.md,
       <out>/panel_few<N>_ratio<R>.png  (rows = test views; columns = GT | full | real-only | one per strategy)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.common import evenly_sample_names, load_split

COLORS = {"inpaint": "#0072B2", "outpaint": "#D55E00", "depth": "#009E73", "edge": "#CC79A7"}  # Okabe-Ito
METRICS = {"PSNR": "PSNR [dB] (higher is better)", "SSIM": "SSIM (higher is better)", "LPIPS": "LPIPS (lower is better)"}


def load_rows(runs: Path) -> list[dict]:
    f = runs / "results.jsonl"
    if not f.exists():
        sys.exit(f"{f} not found; run 05_run_gs_experiments.py first")
    latest = {}
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["name"]] = r  # re-runs overwrite
    return list(latest.values())


def plots(rows: list[dict], out: Path) -> None:
    full = next((r for r in rows if r["name"] == "full"), None)
    for n in sorted({r["n_real"] for r in rows if r["strategy"] != "none" or r["name"] != "full"} - {None}):
        sub = [r for r in rows if r["n_real"] == n and r["name"] != "full"]
        real = next((r for r in sub if r["strategy"] == "none"), None)
        strategies = sorted({r["strategy"] for r in sub if r["strategy"] != "none"})
        if not strategies:
            continue
        for metric, label in METRICS.items():
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            for s in strategies:
                pts = sorted((r["ratio"], r[metric]) for r in sub if r["strategy"] == s)
                if real:
                    pts = [(0, real[metric])] + pts
                ax.plot(*zip(*pts), marker="o", color=COLORS.get(s), label=s)
            if full:
                ax.axhline(full[metric], color="0.4", ls=":", label=f"full-N ({full['n_real']} real)")
            ax.set_xlabel("synthetic : real ratio [%]")
            ax.set_ylabel(label)
            ax.set_title(f"{n}-shot", fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / f"plot_{metric}_few{n}.png", dpi=200)
            plt.close(fig)
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        for ax, key, lab in zip(axs, ["train_time_min", "peak_gpu_mb"], ["training time [min]", "peak GPU memory [MB]"]):
            for s in strategies:
                pts = sorted((r["num_train_images"], r[key]) for r in sub if r["strategy"] == s)
                ax.plot(*zip(*pts), marker="o", color=COLORS.get(s), label=s)
            ax.set_xlabel("training images")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"cost_few{n}.png", dpi=200)
        plt.close(fig)


def table(rows: list[dict], out: Path) -> None:
    hdr = ("| condition | real | synth | time [min] | mem [MB] | PSNR | SSIM | LPIPS | PSNR near-train | PSNR in-gap |\n"
           "|---|---|---|---|---|---|---|---|---|---|\n")
    best = {m: (max if m != "LPIPS" else min)(r[m] for r in rows) for m in METRICS}
    lines = []
    for r in sorted(rows, key=lambda r: (r["n_real"], r["strategy"], r["ratio"])):
        cells = [f"{r[m]:.4f}" if r[m] != best[m] else f"**{r[m]:.4f}**" for m in METRICS]
        lines.append(f"| {r['name']} | {r['n_real']} | {r['num_synth']} | {r['train_time_min']} | {r['peak_gpu_mb']} | "
                     + " | ".join(cells)
                     + f" | {r.get('PSNR_out', float('nan')):.3f} | {r.get('PSNR_gap', float('nan')):.3f} |")
    (out / "results_table.md").write_text(hdr + "\n".join(lines) + "\n")


def render_dir(runs: Path, name: str) -> Path | None:
    d = sorted((runs / name / "test").glob("ours_*"))
    return d[-1] if d else None


def per_view_psnr(runs: Path, name: str) -> dict:
    f = runs / name / "per_view.json"
    try:
        data = json.loads(f.read_text())
        return next(iter(data.values())).get("PSNR", {})
    except Exception:
        return {}


def panels(rows: list[dict], runs: Path, out: Path, root: Path, n_views: int, ratio: int | None) -> None:
    test_sorted = sorted(load_split(root)["test"])  # GS renders test cameras in sorted-name order
    idx = [test_sorted.index(n) for n in evenly_sample_names(test_sorted, n_views)]
    names = {r["name"]: r for r in rows}
    for n in sorted({r["n_real"] for r in rows if r["name"] != "full"}):
        ratios = sorted({r["ratio"] for r in rows if r["n_real"] == n and r["strategy"] != "none"})
        for ratio_ in ([ratio] if ratio else ratios):
            cols = [("ground truth", None, None)]
            if "full" in names:
                cols.append(("full-N", "full", None))
            if f"few{n}_real" in names:
                cols.append((f"{n}-shot real", f"few{n}_real", None))
            for s in sorted({r["strategy"] for r in rows if r["strategy"] != "none"}):
                if f"few{n}_{s}_{ratio_}" in names:
                    cols.append((f"{s} {ratio_}%", f"few{n}_{s}_{ratio_}", None))
            if len(cols) < 3:
                continue
            fig, axs = plt.subplots(len(idx), len(cols), figsize=(2.6 * len(cols), 1.6 * len(idx) + 0.5), squeeze=False)
            for j, (title, cond, _) in enumerate(cols):
                d = render_dir(runs, cond or f"few{n}_real")
                psnr = per_view_psnr(runs, cond) if cond else {}
                for i, k in enumerate(idx):
                    ax = axs[i][j]
                    ax.axis("off")
                    if d is None:
                        continue
                    sub, fname = ("gt" if cond is None else "renders"), f"{k:05d}.png"
                    p = d / sub / fname
                    if p.exists():
                        ax.imshow(Image.open(p))
                    if i == 0:
                        ax.set_title(title, fontsize=8)
                    if fname in psnr:
                        ax.text(0.02, 0.03, f"{psnr[fname]:.1f} dB", transform=ax.transAxes, fontsize=7, color="w",
                                bbox=dict(facecolor="k", alpha=0.5, pad=1, lw=0))
            fig.subplots_adjust(wspace=0.02, hspace=0.02)
            fig.savefig(out / f"panel_few{n}_ratio{ratio_}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/outputs"))
    ap.add_argument("--runs", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--views", type=int, default=3, help="test views per panel")
    ap.add_argument("--panel-ratio", type=int, default=None, help="one ratio only (default: a panel for every ratio)")
    args = ap.parse_args()
    runs = args.runs or args.root / "gs_runs"
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(runs)
    plots(rows, args.out)
    table(rows, args.out)
    panels(rows, runs, args.out, args.root, args.views, args.panel_ratio)
    print(f"wrote plots, table and panels to {args.out}")


if __name__ == "__main__":
    main()
