"""Wait for both selected training runs, then generate AlpacaEval answers."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/parameter-sweep-20261121-143715'
OUT = ROOT / 'outputs/alpaca-eval-20261121'
ARMS = ['lambda-01', 'kappa-rejected-15']
OUT.mkdir(parents=True, exist_ok=True)


def status(stage, **extra):
    data = {'stage': stage, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **extra}
    temp = OUT / 'generation-status.tmp'
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(OUT / 'generation-status.json')
    print(json.dumps(data), flush=True)


def ready():
    return all((RUN / a / 'checkpoint-3753/adapter_config.json').is_file()
               and json.loads((RUN / f'{a}-progress.json').read_text())['step'] == 3753
               for a in ARMS)


with (OUT / 'generation.lock').open('w') as own_lock:
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status('waiting_for_training', target_steps=3753)
    while not ready():
        time.sleep(60)
    # The trainer holds this lock until all final checks and saves finish.
    with (RUN / 'run.lock').open('a') as train_lock:
        fcntl.flock(train_lock, fcntl.LOCK_EX)
        assert ready()
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false')
        models = [('dpo-3753', ROOT / 'outputs/dpo-3b-run1/checkpoint-3753')]
        models += [(a + '-3753', RUN / a / 'checkpoint-3753') for a in ARMS]
        for name, adapter in models:
            status('generating', model=name)
            cmd = [sys.executable, str(ROOT / 'tools/generate_alpaca_outputs.py'),
                   '--base', '/root/autodl-tmp/models/Qwen2.5-3B-Instruct',
                   '--adapter', str(adapter), '--dataset', str(OUT / 'alpaca_eval.json'),
                   '--name', name, '--output', str(OUT / (name + '.json'))]
            with (OUT / (name + '.log')).open('a') as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        status('answers_complete_awaiting_judge', questions_per_model=805)
