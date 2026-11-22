"""Generate resumable AlpacaEval answers from a local base model and LoRA adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-new-tokens', type=int, default=2048)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = json.loads(args.dataset.read_text())
    if args.limit:
        rows = rows[:args.limit]
    assert rows and all(isinstance(r['instruction'], str) for r in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal = args.output.with_suffix('.jsonl')
    meta_path = args.output.with_suffix('.metadata.json')
    metadata = {
        'base': str(Path(args.base).resolve()),
        'adapter': str(Path(args.adapter).resolve()),
        'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        'generator': args.name, 'pairs': len(rows),
        'do_sample': False, 'max_new_tokens': args.max_new_tokens,
        'batch_size': args.batch_size,
        'prompt': 'tokenizer.apply_chat_template, user message, add_generation_prompt=True',
    }
    if meta_path.exists():
        assert json.loads(meta_path.read_text()) == metadata, 'Generation settings changed'
    else:
        meta_path.write_text(json.dumps(metadata, indent=2))
    done = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    assert len(done) <= len(rows)
    assert all(r['instruction'] == rows[i]['instruction'] and r['generator'] == args.name
               for i, r in enumerate(done)), 'Existing outputs do not match dataset'
    if len(done) < len(rows):
        torch.manual_seed(42)
        tok = AutoTokenizer.from_pretrained(args.base, padding_side='left', local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map='cuda',
            attn_implementation='sdpa', local_files_only=True)
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
        model.eval()
        for start in range(len(done), len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = [tok.apply_chat_template([{'role': 'user', 'content': r['instruction']}],
                        tokenize=False, add_generation_prompt=True) for r in batch]
            inputs = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
            assert inputs.input_ids.shape[1] + args.max_new_tokens <= model.config.max_position_embeddings
            with torch.inference_mode():
                seq = model.generate(**inputs, do_sample=False, num_beams=1,
                                     max_new_tokens=args.max_new_tokens,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            generated = seq[:, inputs.input_ids.shape[1]:]
            with journal.open('a', encoding='utf-8') as handle:
                for row, tokens in zip(batch, generated, strict=True):
                    record = {
                        'instruction': row['instruction'],
                        'output': tok.decode(tokens, skip_special_tokens=True),
                        'generator': args.name,
                        'dataset': row.get('dataset', 'alpaca_eval'),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                    done.append(record)
                handle.flush()
                os.fsync(handle.fileno())
            print(f'{args.name}: {len(done)}/{len(rows)}', flush=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(args.output)
    print(f'Complete: {args.output}', flush=True)


if __name__ == '__main__':
    main()
