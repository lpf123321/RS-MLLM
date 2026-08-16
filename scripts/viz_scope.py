#!/usr/bin/env python3
"""
可视化 SCOPE 剪枝位置：随机抽取样本，显示哪些 2x2 spatial block 被保留/剪掉。
叠加在原图上生成热力图。
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import PeftModel

MODEL_PATH = '/home/u2024311149/models/Qwen3.5-4B'
LORA_PATH = '/users/u2024311136/shared/shared_models/sft_stage1_lora'
DATA_ROOT = os.environ.get('DATA_ROOT', '/home/u2024311149/RS-MLLM/datasets/shared_datasets')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prune/output/viz')
os.makedirs(OUT_DIR, exist_ok=True)

from prune.scope import SCOPE_L2, enable_scope_hooks

# ── Load model ──
print("[1/4] Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor.tokenizer.padding_side = 'left'
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()
model.eval()

enable_scope_hooks(model.model.visual)
print("  Scope hook registered on vision block 22")

# ── Load random MME samples ──
print("[2/4] Loading random MME samples...")
all_samples = []
with open(f'{DATA_ROOT}/MME-RealWorld-RS/mme_rs.jsonl') as f:
    for line in f:
        all_samples.append(json.loads(line))
random.seed(42)
random.shuffle(all_samples)

NUM = 6
selected = all_samples[:NUM]
print(f"  Selected {NUM} random samples")

# ── Process each sample ──
print(f"[3/4] Running SCOPE visualization on {NUM} samples...")
results = []

for idx, raw in enumerate(selected):
    msgs = raw['messages']
    if isinstance(msgs, str): msgs = eval(msgs)
    uc = msgs[0]['content']
    text = next(c['text'] for c in uc if c['type'] == 'text')
    imgs = [c['image'] for c in uc if c['type'] == 'image']
    if not imgs:
        continue
    img_path = imgs[0]
    
    print(f"  [{idx}] {os.path.basename(img_path)[:40]}...")
    
    # Build prompt
    msg = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": text},
    ]}]
    prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    
    img_inputs, _ = process_vision_info(msg)
    inputs = processor(text=[prompt], images=img_inputs, padding=True, return_tensors='pt')
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
    
    # Run generation (this triggers vision encoder + SCOPE)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=None, top_p=None)
    
    input_len = inputs['input_ids'].shape[1]
    import re
    output = processor.decode(gen[0, input_len:], skip_special_tokens=True)
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
    
    # Now reconstruct SCOPE selection from features
    # We need to capture grid_thw during the forward — use a wrapper
    # Actually, let's just run the vision encoder separately to get the features
    # First, get grid_thw from the inputs
    grid_thw = inputs.get('image_grid_thw')
    if grid_thw is None:
        print(f"    WARNING: no image_grid_thw, skipping")
        results.append({'idx': idx, 'error': 'no grid_thw'})
        continue
    
    # Run vision encoder directly to get raw features without SCOPE
    # (the hook already fires, but we clean and re-run)
    model.model.visual._scope_features = None
    
    pixel_values = inputs['pixel_values']
    
    vision_out = model.model.visual(pixel_values, grid_thw=grid_thw)
    scope_feat = model.model.visual._scope_features
    
    if scope_feat is None:
        print(f"    WARNING: no _scope_features")
        results.append({'idx': idx, 'error': 'no _scope_features'})
        continue
    
    # Parse grid
    n_raw = scope_feat.shape[0]
    n_merged = int(grid_thw[0, 1].item() * grid_thw[0, 2].item() // 4)
    h_blocks = int(grid_thw[0, 1].item() // 2)
    w_blocks = int(grid_thw[0, 2].item() // 2)
    
    print(f"    raw={n_raw} merged={n_merged} grid={h_blocks}x{w_blocks}")
    
    # Group into blocks and run SCOPE for different R values
    block_feat = scope_feat[:n_merged * 4].view(n_merged, 4, -1).mean(dim=1)  # [n_merged, 1024]
    
    for R in [0.1, 0.2, 0.3, 0.5]:
        num_keep = max(1, int(n_merged * (1.0 - R)))
        sel_blk, _ = SCOPE_L2(block_feat.unsqueeze(0), num_keep)
        selected = torch.zeros(n_merged, dtype=torch.bool)
        selected[sel_blk[0]] = True
        
        # Reshape selected mask to 2D grid
        selected_grid = selected.view(h_blocks, w_blocks).cpu().numpy()
        
        # Load original image
        img_pil = Image.open(img_path).convert('RGB')
        img_w, img_h = img_pil.size
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Left: original image
        axes[0].imshow(img_pil)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Middle: pruning heatmap (red=pruned, green=kept)
        heatmap = np.zeros((h_blocks, w_blocks, 3))
        for hb in range(h_blocks):
            for wb in range(w_blocks):
                if selected_grid[hb, wb]:
                    heatmap[hb, wb] = [0.2, 0.8, 0.2]  # green = kept
                else:
                    heatmap[hb, wb] = [0.9, 0.2, 0.2]  # red = pruned
        
        axes[1].imshow(heatmap, interpolation='nearest')
        axes[1].set_title(f'SCOPE R={R}: Kept (green) vs Pruned (red)\n{h_blocks}x{w_blocks} blocks, {selected.sum().item()}/{n_merged} kept')
        axes[1].axis('off')
        
        # Right: overlay
        overlay = img_pil.resize((w_blocks * 10, h_blocks * 10), Image.NEAREST)
        overlay_arr = np.array(overlay).astype(float) / 255.0
        for hb in range(h_blocks):
            for wb in range(w_blocks):
                if not selected_grid[hb, wb]:
                    overlay_arr[hb*10:(hb+1)*10, wb*10:(wb+1)*10, 0] = 0.8
                    overlay_arr[hb*10:(hb+1)*10, wb*10:(wb+1)*10, 1] *= 0.3
                    overlay_arr[hb*10:(hb+1)*10, wb*10:(wb+1)*10, 2] *= 0.3
        
        axes[2].imshow(overlay_arr)
        axes[2].set_title('Overlay: Red tint = pruned region')
        axes[2].axis('off')
        
        plt.suptitle(f'Sample {idx}: {text[:80]}...\nModel output: {output[:120]}', fontsize=8)
        plt.tight_layout()
        
        fname = f'{OUT_DIR}/sample{idx}_R{R*100:.0f}.png'
        plt.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"    Saved R={R}: {fname}")
    
    results.append({
        'idx': idx, 'text': text[:80], 'output': output[:120],
        'n_merged': n_merged, 'h_blocks': h_blocks, 'w_blocks': w_blocks,
        'img_path': img_path,
    })

# ── Save metadata ──
print(f"\n[4/4] Saving results")
with open(f'{OUT_DIR}/metadata.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Done! Visualizations saved to {OUT_DIR}/")
