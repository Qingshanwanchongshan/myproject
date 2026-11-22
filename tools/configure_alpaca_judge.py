"""Interactively store local judge settings without echoing the API key."""
import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replace', action='store_true', help='Replace the saved key after new input')
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / '.alpaca-private/judge.json'
    if path.exists() and not args.replace:
        raise SystemExit('Config already exists. Use --replace to enter a replacement key.')
    provider = 'OpenAI'
    base_url = 'https://api.openai.com/v1'
    model = 'gpt-4.1-2025-04-14'
    print(f'Provider: {provider}; judge: {model}')
    key = getpass.getpass('API key (hidden): ').strip()
    if not key:
        raise SystemExit('No key supplied. Nothing saved.')
    if not key.isascii() or any(c.isspace() for c in key) or key[0] in '\"\'':
        raise SystemExit('Paste only the API key, without quotes or spaces. Nothing saved.')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if os.name == 'nt':
        account = os.environ['USERDOMAIN'] + '\\' + os.environ['USERNAME']
        subprocess.run(['icacls', str(path.parent), '/inheritance:r', '/grant:r',
                        account + ':(OI)(CI)F'], check=True, capture_output=True)
    fd, temporary = tempfile.mkstemp(prefix='judge-', suffix='.tmp', dir=path.parent)
    os.chmod(temporary, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump({'provider': provider, 'base_url': base_url, 'model': model,
                   'api_key': key}, f)
    os.replace(temporary, path)
    print(f'Saved privately to {path}. Key was not printed.')
    print('The prepared scoring queue will check connectivity and model access first.')


if __name__ == '__main__':
    main()
