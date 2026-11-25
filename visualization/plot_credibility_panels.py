#!/usr/bin/env python3
"""
Side-by-side credibility figure for the PARAGraph paper.

  (a) Lesion faithfulness (D1): expected grade vs fraction of lesions deleted,
      importance order vs random order -- the gap between them is faithfulness.
      Needs interpret_lesion_faithfulness.py's faithfulness.json.
  (b) Segmentation-degradation robustness (C2): test QWK vs degradation rate for
      the three injected error modes (missed / misclassified / false-positive
      lesions). Needs robustness_seg_degradation.py's degradation_*.json.

Both panels share the theme "perturb the lesion evidence, watch the prediction".

Usage:
  python visualization/plot_credibility_panels.py \
      --faith interpret_out/faithfulness.json \
      --degrade robustness_ddr/degradation_paragraph_ddr_run1.json \
      --out figures/credibility_panels
  # preview panel (b) only while D1 has not been (re)run:
  python visualization/plot_credibility_panels.py --degrade robustness_ddr/degradation_paragraph_ddr_run1.json --c2_only --out figures/c2_preview

Colors use the dataviz skill's pre-validated categorical hexes in order
(blue / orange / violet); the JS validator was not run here (no node runtime),
but these are the skill's validated reference slots.
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import glob as _glob
for _f in _glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures/fonts/*.ttf')):
    try:
        _fm.fontManager.addfont(_f)
    except Exception:
        pass

# --- dataviz skill: pre-validated categorical slots (fixed order) + neutral ink
BLUE, ORANGE, VIOLET, GREEN = '#2a78d6', '#eb6834', '#6250d6', '#008300'
INK, MUTED, GRID = '#222222', '#6a6a6a', '#dddddd'

plt.rcParams.update({
    'font.size': 12.5, 'axes.titlesize': 13.5, 'axes.labelsize': 12.5,
    'legend.fontsize': 11, 'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.labelcolor': INK, 'text.color': INK,
    'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
})


def _style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', color=GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def panel_faithfulness(ax, faith, show_text=True):
    grid = np.asarray(faith['grid'])
    imp = np.asarray(faith['deletion_importance'])
    rnd = np.asarray(faith['deletion_random'])
    ins = np.asarray(faith.get('insertion_importance') or [])
    # label offsets proportional to the data range (works for any E[grade] scale)
    lo = min(imp.min(), rnd.min(), ins.min() if ins.size else imp.min())
    hi = max(imp.max(), rnd.max(), ins.max() if ins.size else rnd.max())
    off = 0.065 * (hi - lo)
    # faithfulness = area between the two deletion curves (random stays high,
    # importance drops fast); shade it.
    ax.fill_between(grid, imp, rnd, color=BLUE, alpha=0.10, linewidth=0, zorder=1)
    ax.plot(grid, rnd, color=MUTED, lw=2, ls='--', marker='o', ms=4,
            markevery=2, zorder=2)
    ax.plot(grid, imp, color=BLUE, lw=2, marker='o', ms=4.5,
            markevery=2, zorder=3)
    if ins.size:
        ax.plot(grid, ins, color=GREEN, lw=2, marker='^', ms=4,
                markevery=2, zorder=2)
        ax.text(0.70, np.interp(0.70, grid, ins) + off, 'Insertion',
                color=GREEN, ha='center', fontsize=11.5)
        # random label goes inside the gap so it clears the insertion plateau
        ax.text(0.33, np.interp(0.33, grid, rnd) - 1.6 * off, 'Random',
                color=MUTED, ha='center', fontsize=11.5)
    else:
        ax.text(0.30, np.interp(0.30, grid, rnd) + off, 'Random',
                color=MUTED, ha='center', fontsize=11.5)
    ax.text(0.52, np.interp(0.52, grid, imp) - 1.6 * off, 'Importance',
            color=BLUE, ha='center', fontsize=11.5)
    auc = faith.get('deletion_gap_auc')
    if show_text and auc is not None:
        ax.text(0.97, 0.96, f'faithfulness (gap AUC) = {auc:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=11.5,
                color=INK, bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                     ec=GRID, lw=0.8))
    ax.set_xlabel('Fraction deleted / inserted' if ins.size else 'Fraction deleted')
    ax.set_ylabel('E[grade]')
    ax.set_title('(a) Faithfulness', loc='left', color=INK)
    ax.set_xlim(0, 1)
    _style(ax)


def panel_degradation(ax, deg, show_text=True):
    rates = np.asarray(deg['rates'])
    clean = deg['clean_qwk']
    curves = deg['curves']
    specs = [
        ('missed', 'Missed', BLUE, 'o'),
        ('misclassify', 'Misclassified', ORANGE, 's'),
        ('false_positive', 'False positive', VIOLET, '^'),
    ]
    ax.axhline(clean, color=MUTED, lw=1.0, ls=':', zorder=1)
    if show_text:
        ax.text(0.015, clean, f'clean QWK {clean:.3f}', va='bottom', ha='left',
                fontsize=11, color=MUTED)
    for key, label, color, mk in specs:
        if key not in curves:
            continue
        ys = [curves[key][f'{r:.1f}'] if f'{r:.1f}' in curves[key]
              else curves[key][str(r)] for r in rates]
        ax.plot(rates, ys, color=color, lw=2, marker=mk, ms=4.5, zorder=3)
        if key == 'missed':
            ax.text(0.42, np.interp(0.45, rates, ys) - 0.035, 'Missed',
                    color=color, ha='center', fontsize=11.5)
        elif key == 'misclassify':
            ax.text(0.72, np.interp(0.72, rates, ys) - 0.028, 'Misclassified',
                    color=color, ha='center', fontsize=11.5)
        elif key == 'false_positive':
            ax.text(0.45, np.interp(0.45, rates, ys) + 0.012, 'False positive',
                    color=color, ha='center', fontsize=11.5)
        # direct end-label the steepest (missed) drop
        if show_text and key == 'missed':
            ax.annotate(f'{ys[-1]:.3f}', xy=(rates[-1], ys[-1]),
                        xytext=(-4, -12), textcoords='offset points',
                        fontsize=12, color=color, ha='right')
    ax.set_xlabel('Degradation rate')
    ax.set_ylabel('QWK')
    ax.set_title('(b) Seg. robustness', loc='left', color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(None, clean + 0.035)
    _style(ax)


def main(args):
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    deg = json.load(open(args.degrade)) if args.degrade else None
    faith = json.load(open(args.faith)) if (args.faith and not args.c2_only) else None

    show_text = not args.no_text
    if args.c2_only or faith is None:
        fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.0))
        panel_degradation(ax, deg, show_text=show_text)
        if not args.c2_only:
            print("[warn] no faithfulness.json -> rendered panel (b) only. "
                  "Rerun interpret_lesion_faithfulness.py and pass --faith to get both panels.")
    else:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.0))
        panel_faithfulness(axL, faith, show_text=show_text)
        panel_degradation(axR, deg, show_text=show_text)

    fig.tight_layout(pad=0.4, w_pad=1.0)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', dpi=300, bbox_inches='tight')
    print(f"saved -> {args.out}.pdf / .png")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--faith', default='interpret_out/faithfulness.json')
    ap.add_argument('--degrade', default='robustness_ddr/degradation_paragraph_ddr_run1.json')
    ap.add_argument('--c2_only', action='store_true', help='render only panel (b)')
    ap.add_argument('--no_text', action='store_true',
                    help='suppress in-figure numeric text (gap-AUC box, clean-QWK label, end-point value)')
    ap.add_argument('--out', default='figures/credibility_panels')
    main(ap.parse_args())
