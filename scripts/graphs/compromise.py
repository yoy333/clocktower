"""Final-day belief vs final-day vote as a density plot, optionally split by good's model or brand.

x: good seats' average probability on the real Demon in the final day's pre-nomination probe.
y: share of good players' votes on the Demon's nominations that day that were yes.
If the Demon wasn't nominated on the final day, y is 0. Games without probes, and good wins with
no final-day vote on the Demon (in practice Slayer wins), are left out.
Saturation is how many games land in a cell; hue is good's win rate there.

    python scripts/graphs/compromise.py [GAME_DIRS ...] [--by-model | --by-brand] [--out compromise.png]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from clocktower.analyze import game_metrics  # noqa: E402


def points(dirs):
    pts, skipped, unvoted = [], 0, 0
    for d in dirs:
        m = game_metrics(d)
        if not m or m['winner'] not in ('good', 'evil'):
            continue
        if not m['demon_by_day']:
            skipped += 1
            continue
        final = max(m['demon_by_day'])  # result 'days' counts the night a game ended on
        if m['winner'] == 'good' and final not in m['yes_on_demon_by_day']:
            unvoted += 1
            continue
        x, y = m['demon_by_day'][final], m['yes_on_demon_by_day'].get(final, 0.0)
        pts.append((x, y, m['winner'], m['good_model']))
    return pts, skipped, unvoted


XB, YB = np.linspace(0, 1, 11), np.linspace(-0.05, 1.05, 12)
# hue runs red (evil won) through magenta to blue (good won)
CMAP = LinearSegmentedColormap.from_list('win', [hsv_to_rgb((1 - h / 3, 1, 0.9)) for h in np.linspace(0, 1, 32)])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('games', nargs='*')
    ap.add_argument('--out')
    ap.add_argument('--by-model', action='store_true', help="one panel per good team's model")
    ap.add_argument('--by-brand', action='store_true', help="one panel each for Anthropic and OpenAI good teams")
    return ap.parse_args()


def group(pts, by_model, by_brand):
    if not (by_model or by_brand):
        return {'all': pts}
    key = (lambda m: 'Anthropic' if m.startswith('claude') else 'OpenAI') if by_brand else (lambda m: m)
    groups = {}
    for p in pts:
        groups.setdefault(key(p[3]), []).append(p)
    return dict(sorted(groups.items()))


def draw(ax, pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    total, *_ = np.histogram2d(xs, ys, bins=[XB, YB])
    good, *_ = np.histogram2d(xs, ys, bins=[XB, YB], weights=[p[2] == 'good' for p in pts])
    rate = np.divide(good, total, out=np.zeros_like(total), where=total > 0)
    hsv = np.stack([1 - rate / 3, np.sqrt(total / total.max()), np.full_like(rate, 0.9)], axis=-1)
    hsv[total == 0] = (0, 0, 1)
    ax.imshow(hsv_to_rgb(hsv).transpose(1, 0, 2), origin='lower', extent=(XB[0], XB[-1], YB[0], YB[-1]), aspect='auto')
    for i, j in zip(*np.nonzero(total)):
        ax.text((XB[i] + XB[i + 1]) / 2, (YB[j] + YB[j + 1]) / 2, int(total[i, j]),
                ha='center', va='center', fontsize=7)


def figure(groups, title):
    cols = min(len(groups), 4)
    rows = -(-len(groups) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols + 1.5, 4 * rows + 0.5), squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    for ax, (name, sel) in zip(axes.flat, groups.items()):
        draw(ax, sel)
        ax.set_title(f'{len(sel)} games' if name == 'all' else f'{name} ({len(sel)} games)')
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Good's belief in the real Demon, final day (probe)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Good's yes votes on the Demon, final day")
    fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP), ax=axes, label='good win rate (saturation = games)')
    fig.suptitle(title)
    return fig


def main():
    args = parse_args()
    dirs = args.games or [p.parent for p in sorted((ROOT / 'games').glob('v*/*/game-*/probes.jsonl'))]
    pts, skipped, unvoted = points(dirs)
    print(f'{len(pts)} games plotted, {skipped} without probes and {unvoted} good wins with no final-day vote on the Demon left out')
    suffix = ', by good brand' if args.by_brand else ', by good model' if args.by_model else ''
    fig = figure(group(pts, args.by_model, args.by_brand), 'Final day: belief vs vote' + suffix)
    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches='tight')
        print(f'saved {args.out}')
    else:
        plt.show()

if __name__ == '__main__':
    main()
