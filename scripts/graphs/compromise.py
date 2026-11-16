"""Final-day belief vs final-day vote, one dot per game.

x: good seats' average probability on the real Demon in the final day's pre-nomination probe.
y: share of good players' votes on the Demon's nominations that day that were yes.
If the Demon wasn't nominated on the final day, y is 0. Games without probes are left out.

    python scripts/graphs/compromise.py [GAME_DIRS ...] [--out compromise.png]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from clocktower.analyze import game_metrics  # noqa: E402


def points(dirs):
    pts, skipped = [], 0
    for d in dirs:
        m = game_metrics(d)
        if not m or m['winner'] not in ('good', 'evil'):
            continue
        if not m['demon_by_day']:
            skipped += 1
            continue
        final = max(m['demon_by_day'])  # result 'days' counts the night a game ended on
        x, y = m['demon_by_day'][final], m['yes_on_demon_by_day'].get(final, 0.0)
        pts.append((x, y, m['winner']))
    return pts, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('games', nargs='*')
    ap.add_argument('--out')
    args = ap.parse_args()
    dirs = args.games or [p.parent for p in sorted((ROOT / 'games').glob('v*/*/game-*/probes.jsonl'))]
    pts, skipped = points(dirs)
    print(f'{len(pts)} games plotted, {skipped} without probes left out')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True, constrained_layout=True)
    bins = [np.linspace(0, 1, 11), np.linspace(-0.05, 1.05, 12)]
    panels = (('good', 'Blues', 'good won'), ('evil', 'Oranges', 'evil won'))
    for ax, (outcome, cmap, title) in zip(axes, panels):
        sel = [(x, y) for x, y, o in pts if o == outcome]
        *_, img = ax.hist2d([x for x, _ in sel], [y for _, y in sel], bins=bins, cmap=cmap, cmin=1)
        fig.colorbar(img, ax=ax, label='games')
        ax.set_title(f'{title} ({len(sel)} games)')
        ax.set_xlabel("Good's belief in the real Demon, final day (probe)")
    axes[0].set_ylabel("Good's yes votes on the Demon, final day")
    fig.suptitle('Final day: belief vs vote')
    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches='tight')
        print(f'saved {args.out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
