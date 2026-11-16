"""API cost and tokens by model, per seat, from each game's calls.jsonl.

A model's cost and tokens are divided by the seats it played (seat-games), and cost also by seats
times game length (seat-days). Belief probes are left out (telemetry, not play) unless --probes is given.

    python scripts/cost_by_model.py [GAME_DIRS ...] [--probes]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def game_rows(game_dir, probes):
    """One row per model in the game: cost, input/output tokens, seats and seat-days."""
    calls = pd.DataFrame([json.loads(l) for l in (game_dir / 'calls.jsonl').read_text().splitlines() if l.strip()])
    if calls.empty:
        return pd.DataFrame()
    if not probes and 'kind' in calls:
        calls = calls[calls['kind'] != 'probe']
    usage = pd.json_normalize(calls['usage'].fillna({}).tolist()).reindex(
        columns=['input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens', 'output_tokens']).fillna(0)
    per = pd.DataFrame({
        'model': calls['model'].fillna('?').values,
        'cost': calls['cost'].fillna(0).values,
        'input': usage[['input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens']].sum(axis=1).values,
        'output': usage['output_tokens'].values,
    }).groupby('model').sum()

    result = json.loads((game_dir / 'result.json').read_text())
    seats = Counter((result.get('models') or {}).values())
    per['seats'] = [seats[m] for m in per.index]
    per['seat_days'] = per['seats'] * (result.get('days') or 0)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('games', nargs='*')
    ap.add_argument('--probes', action='store_true', help='include belief-probe calls')
    args = ap.parse_args()
    dirs = [Path(d) for d in args.games] or [p.parent for p in sorted((ROOT / 'games').glob('v*/*/game-*/calls.jsonl'))]
    dirs = [d for d in dirs if (d / 'calls.jsonl').exists() and (d / 'result.json').exists()]

    t = pd.concat([game_rows(d, args.probes) for d in dirs]).groupby(level=0).sum()
    g, n = t['seats'].replace(0, float('nan')), t['seat_days'].replace(0, float('nan'))
    out = pd.DataFrame({
        'seats': t['seats'],
        '$/game': (t['cost'] / g).round(2),
        '$/day': (t['cost'] / n).round(2),
        'in/game': (t['input'] / g).round(0),
        'out/game': (t['output'] / g).round(0),
    }).sort_values('$/game', ascending=False)

    print(f"{len(dirs)} games{' (probes included)' if args.probes else ''}; per seat; input includes cache, output includes thinking\n")
    print(out.to_string(formatters={'in/game': '{:,.0f}'.format, 'out/game': '{:,.0f}'.format}))


if __name__ == '__main__':
    main()
