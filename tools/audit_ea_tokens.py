"""Offline EA localization audit; annotations are synthetic, not causal proofs."""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def dump(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def samples(root):
    from datasets import load_from_disk
    rng = random.Random(20261124)
    rows = []
    for i in range(300):
        a, b = rng.randint(10, 199), rng.randint(10, 199)
        op = ['+', '-', '*'][i % 3]
        result = {'+': a+b, '-': a-b, '*': a*b}[op]
        prefix = ['The answer is ', 'Result: ', 'Calculating this gives ', 'Certainly! The result is ', 'After calculation, the answer is '][i % 5]
        rows.append(dict(id=i, source='synthetic_minimal_arithmetic', prompt=f'What is {a} {op} {b}? Give the result.',
                         chosen=prefix+str(result)+'.', rejected=prefix+str(result+1)+'.', gold_start=len(prefix),
                         gold_end_chosen=len(prefix)+len(str(result)), gold_end_rejected=len(prefix)+len(str(result+1))))
    ds = load_from_disk(str(root/'data/uf-60k-20261122'))['test']
    indices = rng.sample(range(len(ds)), 200)
    for idx in indices:
        r = ds[idx]
        chosen, rejected = r['chosen'], r['rejected']
        if isinstance(chosen, list):
            prompt = [x for x in chosen if x['role'] != 'assistant']
            chosen = [x['content'] for x in chosen if x['role']=='assistant'][-1]
            rejected = [x['content'] for x in rejected if x['role']=='assistant'][-1]
        else:
            prompt = r.get('prompt', '')
        rows.append(dict(id=len(rows), source='natural_validation_unannotated', dataset_index=idx,
                         prompt=prompt, chosen=chosen, rejected=rejected))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    import subprocess
    while True:
        free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
        if free >= 16000:
            break
        dump(out/'status.json', dict(phase='waiting_for_memory', free_mib=free))
        time.sleep(20)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    sys.path.insert(0, str(root/'trl-1.18.1'))
    from trl.trainer.ea_dpo import AttentionCredit, select_attention, token_weights
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.20, 0)
    base = '/data/buaa_hua/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1'
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    rows = samples(root)
    dump(out/'samples.json', rows)
    arms = [('original', 'ea60k-20261122/C-kappa15-lr5e6', 1.5),
            ('tau225', 'ea60k-tau225-20261124/C-tau225-kappa15-lr5e6', 2.25),
            ('tau3', 'ea60k-tau3-20261124/C-tau3-kappa15-lr5e6', 3.)]
    dump(out/'protocol.json', dict(pairs=500, models=3, sequences_per_model=1000, checkpoint=3000,
         annotated_pairs=300, natural_unannotated_pairs=200, seed=20261124,
         limits=['Synthetic arithmetic probes only validate answer-span localization, not broad preference alignment.',
                 'Natural pair labels are response-level and provide no token ground truth.',
                 'No causal intervention performed; attention credit is not causal proof.',
                 'Full logits entropy in FP32 may differ slightly from fused bf16 training entropy.'],
         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    for name, rel, tau in arms:
        journal = out/(name+'.jsonl')
        done = [json.loads(x) for x in journal.read_text(encoding='utf-8').splitlines()] if journal.exists() else []
        completed = {(x['id'],x['side']) for x in done}
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map='cuda',
                                                    attn_implementation='sdpa', local_files_only=True)
        adapter = root/'outputs'/rel/'checkpoint-3000'
        model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True).eval()
        attention, layer = select_attention(model.get_base_model(), 30)
        for row in rows:
            for side in ['chosen','rejected']:
                if (row['id'],side) in completed:
                    continue
                msgs = row['prompt'] if isinstance(row['prompt'],list) else [{'role':'user','content':row['prompt']}]
                prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                prefix_ids = tok.encode(prefix, add_special_tokens=False)
                enc = tok(row[side], add_special_tokens=False, return_offsets_mapping=True)
                ids = prefix_ids + enc.input_ids + [tok.eos_token_id]
                ids = ids[:1024]
                if len(ids)<=len(prefix_ids):
                    rec = dict(id=row['id'],side=side,source=row['source'], skipped='completion_fully_truncated')
                else:
                    x = torch.tensor([ids],device='cuda'); mask=torch.ones_like(x)
                    cm=torch.zeros_like(x); cm[:,len(prefix_ids):]=1
                    with torch.inference_mode():
                        observer=AttentionCredit(attention,cm,mask,128,1e-8)
                        with observer.capture():
                            logits=model(input_ids=x,attention_mask=mask,use_cache=False).logits
                        ent=torch.zeros((1,len(ids)-1),device='cuda')
                        for start in range(0,len(ids)-1,32):
                            lp=logits[:,start:min(start+32,len(ids)-1)].float().log_softmax(-1)
                            ent[:,start:start+lp.shape[1]]=-(lp.exp()*lp).sum(-1)
                        credit=observer.credit[:,1:]; valid=cm[:,1:]
                        weights, details=token_weights(ent.repeat(2,1),credit.repeat(2,1),valid.repeat(2,1),
                            redistribution=.1,tau_chosen=2.,tau_rejected=tau,kappa_chosen=1.,kappa_rejected=1.5,
                            gamma=.1,cap_quantile=.95,epsilon=1e-8,return_details=True)
                        sel=0 if side=='chosen' else 1
                        positions=valid[0].bool()
                        h,c,info,probs=[v[sel,positions].cpu().tolist() for v in details]
                        w=weights[sel,positions].cpu().tolist()
                        raw_h=ent[0,positions].cpu().tolist(); raw_c=credit[0,positions].cpu().tolist()
                    token_ids=ids[len(prefix_ids):]
                    offsets=list(enc.offset_mapping[:len(token_ids)])
                    offsets += [(len(row[side]),len(row[side]))]*(len(token_ids)-len(offsets))
                    gold=[i for i,(a,b) in enumerate(offsets) if 'gold_start' in row and b>row['gold_start'] and a<row['gold_end_'+side]]
                    metrics={}
                    for signal, scores in [('I',info),('h',h),('c',c)]:
                        order=sorted(range(len(scores)),key=lambda i:(-scores[i],i)); metrics[signal]={}
                        for frac in [.1,.3]:
                            k=max(1,math.ceil(len(scores)*frac)); top=order[:k]
                            random_top=random.Random(20261124+row['id']*2+(side=='rejected')).sample(range(len(scores)),k)
                            metrics[signal][str(frac)]=dict(top_indices=top,
                                gold_recall=len(set(top)&set(gold))/len(gold) if gold else None,
                                random_gold_recall=len(set(random_top)&set(gold))/len(gold) if gold else None,
                                random_expected_recall=k/len(scores) if gold else None)
                    mass={str(f):sum(sorted(w,reverse=True)[:max(1,math.ceil(len(w)*f))])/sum(w) for f in [.01,.05,.1,.3]}
                    rec=dict(id=row['id'],side=side,source=row['source'],text=row[side],layer=layer,
                             token_ids=token_ids,tokens=[tok.decode([i]) for i in token_ids],offsets=offsets,gold_indices=gold,
                             raw_entropy=raw_h,raw_credit=raw_c,h=h,c=c,interaction=[.1*a*b for a,b in zip(h,c)],
                             I=info,p=probs,w=w,metrics=metrics,top_weight_mass=mass,
                             effective_token_fraction=sum(w)**2/(len(w)*sum(v*v for v in w)),
                             truncated=len(prefix_ids)+len(enc.input_ids)+1>1024)
                    assert abs(sum(w)/len(w)-1)<1e-4
                    del logits,ent,credit,weights,details
                with journal.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
                done.append(rec)
                dump(out/'status.json',dict(phase='running',model=name,completed_sequences=len(done),total_sequences=1000))
                print(name,len(done),'/1000',flush=True)
        cards=[]
        for r in done:
            if 'skipped' in r: continue
            top=set(r['metrics']['I']['0.1']['top_indices']); pieces=[]; prev=0
            for i,(a,b) in enumerate(r['offsets']):
                if b<=a:continue
                pieces.append(html.escape(r['text'][prev:a]))
                title=html.escape(f"I={r['I'][i]:.3f} h={r['h'][i]:.3f} c={r['c'][i]:.3f} w={r['w'][i]:.3f}")
                pieces.append(f'<mark title="{title}">'+html.escape(r['text'][a:b])+'</mark>' if i in top else html.escape(r['text'][a:b]));prev=b
            pieces.append(html.escape(r['text'][prev:]));cards.append(f"<h3>{r['id']} {r['side']} {r['source']}</h3><pre style='white-space:pre-wrap'>"+''.join(pieces)+'</pre>')
        (out/(name+'.html')).write_text('<meta charset="utf-8"><h1>EA top10% localization audit</h1><p>Hover highlighted tokens for component scores. Not causal proof.</p>'+''.join(cards),encoding='utf-8')
        del model,attention;torch.cuda.empty_cache()
    dump(out/'status.json',dict(phase='complete',models=3,pairs_per_model=500,total_sequences=3000))


if __name__=='__main__':
    main()
