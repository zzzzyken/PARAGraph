#!/usr/bin/env python3
"""
Re-render the IG attribution figures (ig_plots_ddr) from the per-image summary
CSV, styled to match figures/credibility_panels (DejaVu, no bold, despined,
light grid, validated categorical palette, single-hue heatmaps).

Outputs (PNG+PDF):
  ig_bar_clean          3-panel bars: All / Correct / Wrong x node groups
  ig_heatmap_clean      per-GT-grade heatmaps (node groups + lesion categories)
  ig_heatmap_correct_clean   same, restricted to correctly predicted images

Usage:
  python visualization/plot_ig_figures.py --csv ig_plots_ddr/ig_summary_paragraph_20260620_103519_ddr.csv \
      --out_dir ig_plots_ddr
"""
import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, VIOLET, GREEN = '#2a78d6', '#eb6834', '#6250d6', '#008300'
INK, MUTED, GRID = '#222222', '#6a6a6a', '#dddddd'
GROUPS = ['Anatomical', 'Image-Level', 'Categorical', 'Region']
GROUP_SHORT = ['Anat.', 'Image', 'Categ.', 'Region']
GROUP_COLORS = [BLUE, ORANGE, VIOLET, GREEN]
LESIONS = ['MA', 'HE', 'EX', 'SE']

plt.rcParams.update({
    'font.size': 12.5, 'axes.titlesize': 13.5, 'axes.labelsize': 12.5,
    'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.labelcolor': INK, 'text.color': INK,
    'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
})


def load_rows(path):
    with open(path) as f:
        return [{k: (v if k == 'image_id' else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]


def _save(fig, out_base):
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_base}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {out_base}.png/.pdf')


# --------------------------------------------------------------------------- #
def bar_figure(rows, out_base):
    """Single-column layout: one panel, grouped bars (All/Correct/Wrong)."""
    subsets = [('All', rows, BLUE),
               ('Correct', [r for r in rows if r['correct'] == 1], GREEN),
               ('Wrong', [r for r in rows if r['correct'] == 0], ORANGE)]
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.0))
    x = np.arange(len(GROUPS))
    w = 0.26
    for k, (label, sub, color) in enumerate(subsets):
        means = [float(np.mean([r[g] for r in sub])) for g in GROUPS]
        bars = ax.bar(x + (k - 1) * w, means, width=w * 0.92, color=color,
                      label=f'{label} (n={len(sub)})', zorder=3)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=10.5, color=INK)
    ax.set_xticks(x, GROUPS)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel('Mean normalized IG')
    ax.grid(True, axis='y', color=GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=3)
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout(pad=0.4)
    _save(fig, out_base)


# --------------------------------------------------------------------------- #
def heatmap_figure(rows, out_base):
    # group by PREDICTED grade (matches the original figure's semantics:
    # "what the model attributed to when predicting grade X"; for the
    # correct-only variant pred == gt anyway)
    grades = sorted({int(r['pred']) for r in rows})
    hl = np.array([[np.mean([r[g] for r in rows if int(r['pred']) == gr]) for g in GROUPS]
                   for gr in grades])
    cat = np.array([[np.mean([r[l] for r in rows if int(r['pred']) == gr]) for l in LESIONS]
                    for gr in grades])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={'width_ratios': [1, 1]})
    vmax = 0.8
    for ax, mat, cols, title in [
            (axes[0], hl, GROUP_SHORT, '(a) Node groups'),
            (axes[1], cat, LESIONS, '(b) Lesion categories')]:
        im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=vmax, aspect='auto')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=12,
                        color='white' if v > 0.55 * vmax else INK)
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(len(grades)), [f'Grade {g}' for g in grades])
        ax.set_title(title, loc='left', color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[1].set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label('Mean normalized IG', fontsize=12)
    cbar.outline.set_visible(False)
    _save(fig, out_base)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out_dir', default='ig_plots_ddr')
    args = ap.parse_args()
    rows = load_rows(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)
    bar_figure(rows, os.path.join(args.out_dir, 'ig_bar_clean'))
    heatmap_figure(rows, os.path.join(args.out_dir, 'ig_heatmap_clean'))
    heatmap_figure([r for r in rows if r['correct'] == 1],
                   os.path.join(args.out_dir, 'ig_heatmap_correct_clean'))
