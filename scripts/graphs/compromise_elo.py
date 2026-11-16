"""Does voting against your own belief explain a model's detection (good-side) rating?

Per good model, from the same final-day points as compromise.py:
  belief:     mean probability good put on the real Demon.
  compromise: how far good's yes-votes on the Demon fall below what a pooled belief->vote line predicts
              (positive = votes less than it believes).
Then detection Elo (docs/plans/f2-ratings.json) is regressed on belief and on belief + compromise.
Intervals are 90% bootstraps over games.

    python scripts/graphs/compromise_elo.py [GAME_DIRS ...] [--boot 1000] [--out compromise_elo.png]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compromise import ROOT, points  # noqa: E402


def per_model(pts):
    x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    slope, icept = np.polyfit(x, y, 1)
    resid = icept + slope * x - y
    out = {}
    for i, p in enumerate(pts):
        out.setdefault(p[3], ([], []))
        out[p[3]][0].append(x[i])
        out[p[3]][1].append(resid[i])
    return {m: (np.mean(b), np.mean(c), len(b)) for m, (b, c) in out.items()}


def bootstrap(pts, boot, seed=0):
    rng = np.random.default_rng(seed)
    samples = {}
    for _ in range(boot):
        for m, (b, c, _) in per_model([pts[i] for i in rng.integers(len(pts), size=len(pts))]).items():
            samples.setdefault(m, ([], []))
            samples[m][0].append(b)
            samples[m][1].append(c)
    return {m: (np.percentile(b, [5, 95]), np.percentile(c, [5, 95])) for m, (b, c) in samples.items()}


def fit(elo, cols):
    X = np.column_stack([np.ones(len(elo)), *cols])
    coef, *_ = np.linalg.lstsq(X, elo, rcond=None)
    pred = X @ coef
    r2 = 1 - ((elo - pred) ** 2).sum() / ((elo - elo.mean()) ** 2).sum()
    return coef, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('games', nargs='*')
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--out')
    args = ap.parse_args()
    dirs = args.games or [p.parent for p in sorted((ROOT / 'games').glob('v*/*/game-*/probes.jsonl'))]
    pts, *_ = points(dirs)
    ratings = {m['id']: m for m in json.loads((ROOT / 'docs/plans/f2-ratings.json').read_text())['models']}

    stats, ci = per_model(pts), bootstrap(pts, args.boot)
    models = sorted((m for m in stats if m in ratings), key=lambda m: ratings[m]['detection'])
    missing = sorted(set(stats) - set(ratings))
    if missing:
        print(f'no rating for: {", ".join(missing)}')

    print(f'{"model":<18}{"games":>6}{"detection":>11}{"belief":>18}{"compromise":>22}')
    for m in models:
        b, c, n = stats[m]
        (bl, bh), (cl, ch) = ci[m]
        print(f'{m:<18}{n:>6}{ratings[m]["detection"]:>11}'
              f'{b:>7.2f} [{bl:.2f},{bh:.2f}]{c:>+9.3f} [{cl:+.2f},{ch:+.2f}]')

    elo = np.array([ratings[m]['detection'] for m in models], float)
    belief = np.array([stats[m][0] for m in models])
    comp = np.array([stats[m][1] for m in models])
    (_, kb), r2b = fit(elo, [belief])
    (_, kb2, kc), r2bc = fit(elo, [belief, comp])
    print(f'\ndetection ~ belief:              R2 {r2b:.2f}  ({kb / 10:+.0f} Elo per +0.1 belief)')
    print(f'detection ~ belief + compromise: R2 {r2bc:.2f}  ({kb2 / 10:+.0f} per +0.1 belief, '
          f'{kc / 10:+.0f} per +0.1 compromise)')
    print(f'corr(belief, compromise): {np.corrcoef(belief, comp)[0, 1]:+.2f}  ({len(models)} models)')

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    err = np.array([[stats[m][1] - ci[m][1][0], ci[m][1][1] - stats[m][1]] for m in models]).T
    ax.errorbar(comp, elo, xerr=err, fmt='none', ecolor='0.7', zorder=1)
    sc = ax.scatter(comp, elo, c=belief, cmap='viridis', s=60, zorder=2)
    for m, x, y in zip(models, comp, elo):
        ax.annotate(m, (x, y), xytext=(5, 4), textcoords='offset points', fontsize=8)
    fig.colorbar(sc, ax=ax, label='mean final-day belief in the real Demon')
    ax.axvline(0, color='0.85', lw=1, zorder=0)
    ax.set_xlabel('compromise: yes-votes on the Demon below what belief predicts')
    ax.set_ylabel('detection rating (Elo)')
    ax.set_title(f'Compromise vs detection rating (R2 belief {r2b:.2f} -> +compromise {r2bc:.2f})')
    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches='tight')
        print(f'saved {args.out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
