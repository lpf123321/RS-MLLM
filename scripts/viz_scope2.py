#!/usr/bin/env python3
"""SCOPE可视化 v2：对比 baseline vs SCOPE(多R值) 的回答 + 剪枝热力图。"""
import sys, os, json, random, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import PeftModel
from scripts.prune import apply_pruning, enable_pruning
from prune.scope import SCOPE_L2

MODEL_PATH = '/home/u2024311149/models/Qwen3.5-4B'
LORA_PATH = 'M_ROOT/sft_stage1_lora'
DATA_ROOT = os.environ.get('DATA_ROOT', '/home/u2024311149/RS-MLLM/datasets/shared_datasets')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prune/output/viz2')
os.makedirs(OUT_DIR, exist_ok=True)

R_VALUES = [0.1, 0.2, 0.3, 0.5]

# ── Load 8 MME samples ──
print("[1/3] Loading MME samples...")
all_samples = []
with open(f'{DATA_ROOT}/MME-RealWorld-RS/mme_rs.jsonl') as f:
    for line in f:
        all_samples.append(json.loads(line))
random.seed(42); random.shuffle(all_samples)
samples = all_samples[:8]

# ── Load model once ──
print("[2/3] Loading model...")
apply_pruning(model_path=MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor.tokenizer.padding_side = 'left'
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
model = PeftModel.from_pretrained(model, LORA_PATH); model = model.merge_and_unload(); model.eval()

def run_one(backbone, processor, model_obj, raw_sample, r_val=None):
    """Run inference on one sample. If r_val is None = baseline (no patch active)."""
    msgs = raw_sample['messages']
    if isinstance(msgs, str): msgs = eval(msgs)
    uc = msgs[0]['content']
    text = next(c['text'] for c in uc if c['type']=='text')
    imgs = [c['image'] for c in uc if c['type']=='image']
    
    msg = [{"role":"user","content":[
        *[{"type":"image","image":i} for i in imgs],
        {"type":"text","text":text}]}]
    prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    img_in, _ = process_vision_info(msg)
    inputs = processor(text=[prompt], images=img_in, padding=True, return_tensors='pt')
    inputs = {k: v.to(model_obj.device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
    
    # Get grid info
    grid_thw = inputs['image_grid_thw']
    hb = int(grid_thw[0,1].item() // 2)
    wb = int(grid_thw[0,2].item() // 2)
    n_merged = hb * wb
    
    # Clear scope features
    if hasattr(backbone.visual, '_scope_features'):
        backbone.visual._scope_features = None
    
    with torch.no_grad():
        gen = model_obj.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=None, top_p=None)
    i_len = inputs['input_ids'].shape[1]
    out = processor.decode(gen[0, i_len:], skip_special_tokens=True)
    out = re.sub(r'<think>.*?</think>', '', out, flags=re.DOTALL).strip()
    
    # For SCOPE runs, capture the feature mask
    mask = None
    if r_val is not None:
        sf = backbone.visual._scope_features
        if sf is None:
            # Pruning already consumed it; re-run vision encoder
            backbone.visual(inputs['pixel_values'], grid_thw=grid_thw)
            sf = backbone.visual._scope_features
        if sf is not None:
            raw = sf[:n_merged*4]
            bf = raw.view(n_merged, 4, -1).mean(dim=1)
            nk = max(1, int(n_merged*(1.0-r_val)))
            sel, _ = SCOPE_L2(bf.unsqueeze(0), nk)
            selected = torch.zeros(n_merged, dtype=torch.bool, device=sel.device)
            selected[sel[0]] = True
            mask = selected.view(hb, wb).cpu().numpy()
    
    return out, mask

backbone = model.model  # Qwen3_5Model

# ── Baseline ──
print("\n=== BASELINE ===")
baseline_outs = []
for idx, s in enumerate(samples):
    o, _ = run_one(backbone, processor, model, s, r_val=None)
    baseline_outs.append(o)
    print(f"  [{idx}] {o[:80]}")

# ── SCOPE with each R ──
print("\n=== SCOPE ===")
scope_outs = {r: [] for r in R_VALUES}
scope_masks = {r: [] for r in R_VALUES}

# Register hook once
from prune.scope import enable_scope_hooks
enable_scope_hooks(backbone.visual)
backbone._prune_method = 'scope'

for r in R_VALUES:
    backbone._prune_r = r
    print(f"\n  R={r} (keep {(1-r)*100:.0f}%):")
    for idx, s in enumerate(samples):
        o, m = run_one(backbone, processor, model, s, r_val=r)
        scope_outs[r].append(o)
        scope_masks[r].append(m)
        flag = '⚠ SHORT' if len(o.strip()) < 5 else '✓'
        print(f"    [{idx}] {flag} {o[:80]}")

# ── Generate comparison images ──
print("\n[3/3] Generating visualizations...")
for idx, s in enumerate(samples):
    msgs = s['messages']
    if isinstance(msgs, str): msgs = eval(msgs)
    uc = msgs[0]['content']
    text = next(c['text'] for c in uc if c['type']=='text')
    img_path = next(c['image'] for c in uc if c['type']=='image')
    img_pil = Image.open(img_path).convert('RGB')
    
    for r in R_VALUES:
        mask = scope_masks[r][idx]
        if mask is None: continue
        hb, wb = mask.shape
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        
        # Left: overlay
        overlay = np.array(img_pil.resize((wb*16, hb*16), Image.NEAREST)).astype(float)/255.0
        for hi in range(hb):
            for wi in range(wb):
                if not mask[hi, wi]:
                    overlay[hi*16:(hi+1)*16, wi*16:(wi+1)*16, 0] = 0.9
                    overlay[hi*16:(hi+1)*16, wi*16:(wi+1)*16, 1:] *= 0.25
        axes[0].imshow(overlay)
        axes[0].set_title(f'SCOPE R={r}: {mask.sum()}/{hb*wb} blocks kept', fontsize=10)
        axes[0].axis('off')
        
        # Right: text
        axes[1].axis('off')
        bas = baseline_outs[idx]
        sco = scope_outs[r][idx]
        color = '#27ae60' if bas[:5].lower() == sco[:5].lower() else '#e74c3c'
        disp = (f"Q: {text[:150]}\n\n"
                f"Baseline: {bas[:200]}\n\n"
                f"SCOPE R={r}: {sco[:200]}")
        axes[1].text(0.02, 0.95, disp, transform=axes[1].transAxes,
                    fontsize=8, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.95))
        
        plt.suptitle(f'Sample {idx}', fontsize=9)
        plt.tight_layout()
        fname = f'{OUT_DIR}/cmp_{idx}_R{r*100:.0f}.png'
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close()

# ── Summary table ──
print(f"\n{'='*90}")
print(f"{'Idx':4s} {'Baseline':25s} {'R=0.1':25s} {'R=0.2':25s} {'R=0.3':25s}")
print(f"{'─'*4} {'─'*25} {'─'*25} {'─'*25} {'─'*25}")
for idx in range(len(samples)):
    b = baseline_outs[idx][:23].ljust(25)
    s01 = scope_outs[0.1][idx][:23].ljust(25)
    s02 = scope_outs[0.2][idx][:23].ljust(25)
    s03 = scope_outs[0.3][idx][:23].ljust(25)
    print(f"{idx:4d} {b} {s01} {s02} {s03}")

# Check match rate vs baseline
for r in R_VALUES:
    match_count = sum(1 for i in range(len(samples)) 
                      if baseline_outs[i][:10].lower() == scope_outs[r][i][:10].lower())
    short_count = sum(1 for i in range(len(samples)) if len(scope_outs[r][i].strip()) < 5)
    print(f"\nR={r}: {match_count}/{len(samples)} match baseline output; {short_count}/{len(samples)} short outputs")

print(f"\nVisualizations: {OUT_DIR}/cmp_*.png")
print(f"Total: {len([f for f in os.listdir(OUT_DIR) if f.endswith('.png')])} PNG files")
