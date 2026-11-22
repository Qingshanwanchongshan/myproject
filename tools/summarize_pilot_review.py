"""Summarize recorded in-conversation judgments; never label them official AE2 scores."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--comparison', choices=['lambda', 'kappa', 'both'], default='both')
    args = parser.parse_args()
    root = args.directory
    summaries = []
    comparisons = ['lambda', 'kappa'] if args.comparison == 'both' else [args.comparison]
    for tag in comparisons:
        candidate = 'lambda-01' if tag == 'lambda' else 'kappa-rejected-15'
        judgments = json.loads((root / f'review-{tag}-decisions.json').read_text())
        key = json.loads((root / f'review-{tag}-key.json').read_text())
        pairs = json.loads((root / f'review-{tag}-pairs.json').read_text(encoding='utf-8'))
        assert len(judgments) == len(pairs) == 100
        assert sorted(j['id'] for j in judgments) == list(range(100))
        values = []
        for row in judgments:
            assert row['winner'] in ['A', 'B', 'tie']
            model = 'tie' if row['winner'] == 'tie' else key[str(row['id'])][row['winner']]
            row['preferred_model'] = model
            values.append(0.5 if model == 'tie' else float(model == candidate))
        wins = sum(x == 1 for x in values)
        ties = sum(x == 0.5 for x in values)
        losses = sum(x == 0 for x in values)
        mean = statistics.mean(values)
        se = statistics.stdev(values) / math.sqrt(len(values))
        summary = {
            'candidate': candidate + '-3325', 'baseline': 'dpo-3753', 'n': 100,
            'wins': wins, 'losses': losses, 'ties': ties,
            'tie_half_preference_score': mean,
            'approx_95pct_prompt_sampling_interval': [max(0, mean - 1.96 * se), min(1, mean + 1.96 * se)],
            'interval_scope': 'prompt sampling only; excludes judge bias and run-to-run uncertainty',
            'official_alpacaeval_score': False, 'length_controlled_score_computed': False,
            'judgment_file_sha256': hashlib.sha256((root / f'review-{tag}-decisions.json').read_bytes()).hexdigest(),
        }
        (root / f'review-{tag}-decisions-decoded.json').write_text(json.dumps(judgments, indent=2))
        with (root / f'{tag}-review.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'winner', 'preferred_model', 'reason', 'pass'])
            writer.writeheader()
            writer.writerows(judgments)
        summaries.append(summary)
    result = {'method': 'single-pass in-conversation custom judge guided by official quality/relevance prompt',
              'official_leaderboard_result': False,
              'position_swapped_independent_pass': False,
              'comparisons': summaries}
    (root / 'review-summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
