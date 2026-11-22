"""Synchronize generated answers, then score when the user supplies an API key."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiment_records/2026-11-21/alpaca-eval'
REMOTE = '/root/autodl-tmp/EA-DPO/outputs/alpaca-eval-20261121'


def status(stage, **kwargs):
    row = {'stage': stage, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **kwargs}
    temp = OUT / 'scoring-status.tmp'
    temp.write_text(json.dumps(row, indent=2))
    temp.replace(OUT / 'scoring-status.json')
    print(json.dumps(row), flush=True)


def main():
    import paramiko
    password = sys.stdin.readline().rstrip('\r\n')
    if not password:
        raise SystemExit('SSH password must be supplied through stdin.')
    status('waiting_for_remote_answers')
    failures = 0
    while True:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect('connect.bjb2.seetacloud.com', port=54028, username='root',
                           password=password, look_for_keys=False, allow_agent=False,
                           timeout=15, auth_timeout=15, banner_timeout=15)
            with client.open_sftp() as sftp:
                with sftp.open(REMOTE + '/generation-status.json') as handle:
                    generation = json.load(handle)
                if generation['stage'] == 'answers_complete_awaiting_judge':
                    for name in ['dpo-3753', 'lambda-01-3753', 'kappa-rejected-15-3753']:
                        target = OUT / (name + '.json')
                        temp = target.with_suffix('.tmp')
                        sftp.get(REMOTE + '/' + target.name, str(temp))
                        assert len(json.loads(temp.read_text(encoding='utf-8'))) == 805
                        temp.replace(target)
                    break
            failures = 0
        except Exception as exc:
            failures += 1
            status('remote_poll_retry', error_type=type(exc).__name__, failures=failures)
            if failures >= 5:
                raise SystemExit('Remote sync blocked after five attempts; inspect connection.')
        finally:
            client.close()
        time.sleep(60)
    password = None
    status('waiting_for_api_key')
    while not (ROOT / '.alpaca-private/judge.json').is_file():
        time.sleep(30)
    status('scoring')
    python = ROOT / '.alpaca-eval-venv/Scripts/python.exe'
    with (OUT / 'scoring.log').open('a', encoding='utf-8') as log:
        result = subprocess.run([str(python), str(ROOT / 'tools/score_alpaca_outputs.py')],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    status('complete' if result.returncode == 0 else 'scoring_failed', exit_code=result.returncode)


if __name__ == '__main__':
    main()
