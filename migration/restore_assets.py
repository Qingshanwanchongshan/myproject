"""Restore Git LFS assets locally; never transfer through SSH/FRP."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets = root / 'migration/assets'
    manifest = json.loads((assets / 'manifest.json').read_text(encoding='utf-8'))
    sources = {}
    for entry in manifest['downloaded_unique_files']:
        source = assets / 'snapshot' / entry['relative']
        if source.stat().st_size != entry['size'] or sha(source) != entry['sha256']:
            raise SystemExit(f'Asset missing/corrupt or still an LFS pointer: {source}. Run git lfs pull.')
        sources[entry['sha256']] = source
    print(f"Verified {len(sources)} unique assets.")
    if args.verify_only:
        return
    for entry in manifest['files']:
        relative = Path(entry['relative'])
        if relative.parts[0] == 'EA-DPO':
            relative = Path(*relative.parts[1:])
        target = root / relative
        if target.exists():
            if sha(target) != entry['sha256']:
                raise SystemExit(f'Refusing to overwrite a different existing file: {target}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sources[entry['sha256']], target)
    records = root / 'experiment_records/restored-20261122'
    records.mkdir(parents=True, exist_ok=True)
    for name in ['core-results.tar.gz', 'local-records.tar.gz']:
        with tarfile.open(assets / name) as t:
            t.extractall(records, filter='data')
    print('Restored exact datasets, historical adapters, tokenizer files, and records.')
    print('Full Qwen2.5-3B-Instruct BASE WEIGHTS are not included; download from an approved source.')
    print('Optimizer states are not included; this supports fresh training and generation evaluation.')


if __name__ == '__main__':
    main()
