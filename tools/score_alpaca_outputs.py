"""Score all three models with official AlpacaEval logic and an explicitly updated judge."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiment_records/2026-11-21/alpaca-eval'
MODEL = 'gpt-4.1-2025-04-14'
NAMES = ['dpo-3753', 'lambda-01-3753', 'kappa-rejected-15-3753']


def prepare():
    import alpaca_eval
    import yaml
    package = Path(alpaca_eval.__file__).parent
    original = package / 'evaluators_configs/weighted_alpaca_eval_gpt4_turbo/configs.yaml'
    config = yaml.safe_load(original.read_text())
    item = config['weighted_alpaca_eval_gpt4_turbo']
    assert item['completions_kwargs']['model_name'] == 'gpt-4-1106-preview'
    item['prompt_template'] = str(package / 'evaluators_configs' / item['prompt_template'])
    item['completions_kwargs']['model_name'] = MODEL
    item['completions_kwargs']['num_procs'] = 4
    target = OUT / 'annotators/weighted_alpaca_eval_gpt41_snapshot'
    target.mkdir(parents=True, exist_ok=True)
    (target / 'configs.yaml').write_text(yaml.safe_dump({'weighted_alpaca_eval_gpt41_snapshot': item}))
    data = json.loads((OUT / 'alpaca_eval.json').read_text(encoding='utf-8'))
    ref = json.loads((OUT / 'alpaca_eval_gpt4_baseline.json').read_text(encoding='utf-8'))
    assert len(data) == len(ref) == 805
    assert [r['instruction'] for r in data] == [r['instruction'] for r in ref]
    manifest = {
        'evaluation': 'AlpacaEval 2 with updated judge; not the original leaderboard configuration',
        'judge': MODEL, 'reference': 'official gpt4_1106_preview outputs',
        'package_version': alpaca_eval.__version__ if hasattr(alpaca_eval, '__version__') else '0.6.6',
        'original_config_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
        'resource_sha256': {n: hashlib.sha256((OUT / n).read_bytes()).hexdigest()
                           for n in ['alpaca_eval.json', 'alpaca_eval_gpt4_baseline.json', 'df_gamed.csv']},
        'models': NAMES,
    }
    (OUT / 'evaluation-manifest.json').write_text(json.dumps(manifest, indent=2))
    return target


def run():
    secret = ROOT / '.alpaca-private/judge.json'
    if not secret.is_file():
        raise SystemExit('Missing API key. Run tools/configure_alpaca_judge.py locally.')
    credentials = json.loads(secret.read_text())
    assert credentials['base_url'] == 'https://api.openai.com/v1' and credentials['model'] == MODEL
    os.environ['OPENAI_API_KEY'] = credentials['api_key']
    os.environ['OPENAI_API_BASE'] = credentials['base_url']
    os.environ['OPENAI_BASE_URL'] = credentials['base_url']
    config = prepare()
    from openai import OpenAI
    client = OpenAI(api_key=credentials['api_key'], base_url=credentials['base_url'], timeout=20, max_retries=0)
    try:
        result = client.chat.completions.create(model=MODEL,
            messages=[{'role': 'user', 'content': 'Reply with exactly M.'}],
            max_tokens=1, temperature=1, logprobs=True, top_logprobs=5)
        assert result.choices[0].logprobs.content, 'Missing token logprobs'
    except Exception as exc:
        # API exceptions can include response bodies, so do not expose them or credentials.
        raise SystemExit(f'Judge preflight failed: {type(exc).__name__}; HTTP {getattr(exc, "status_code", "unknown")}') from None

    from alpaca_eval.main import evaluate
    import alpaca_eval.metrics.glm_winrate as glm
    hub_download = glm.hf_hub_download

    def cached_download(*args, **kwargs):
        if kwargs.get('repo_id') == 'tatsu-lab/alpaca_eval' and kwargs.get('filename') == 'df_gamed.csv':
            return str(OUT / 'df_gamed.csv')
        return hub_download(*args, **kwargs)

    glm.hf_hub_download = cached_download
    refs = json.loads((OUT / 'alpaca_eval_gpt4_baseline.json').read_text(encoding='utf-8'))
    for name in NAMES:
        rows = json.loads((OUT / (name + '.json')).read_text(encoding='utf-8'))
        assert len(rows) == 805 and [r['instruction'] for r in rows] == [r['instruction'] for r in refs]
        evaluate(model_outputs=rows, reference_outputs=refs, annotators_config=str(config),
                 name=name, output_path=str(OUT / 'scores' / name),
                 precomputed_leaderboard=None,
                 metric_kwargs={'save_weights_dir': str(OUT / 'glm-weights')})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if args.prepare_only:
        prepare()
        print('Official resources and updated-judge configuration verified; no API call made.')
    else:
        run()
