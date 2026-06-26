# explore_remixer.py
#  Stage C: Sound Source Remixer
# Input:  data/clip_embeddings.npy   (19 x 768, from Kavana's CLIP run)
#         data/clip_labels.json      (object labels, from Kavana)
#         data/detections.json       (bounding boxes, from Kruthika)
# Output: data/remixer_output.npy   (19 x 768, attention-blended embeddings)

import json
import numpy as np
import torch
import torch.nn.functional as F

# ── 1. Load all inputs ───────────────────────────────────────────────────────

clip_emb = np.load("data/clip_embeddings.npy")          # (19, 768)
with open("data/clip_labels.json") as f:
    clip_labels = json.load(f)
with open("data/detections.json") as f:
    detections = json.load(f)

print("=== SS2A Stage C: Sound Source Remixer ===\n")
print(f"CLIP embeddings loaded : {clip_emb.shape}")
print(f"Objects detected       : {len(detections)}")
print(f"Object labels          : {[d['label'] if 'label' in d else list(d.values())[0] for d in detections[:5]]} ...")

# ── 2. Convert to torch ──────────────────────────────────────────────────────

emb = torch.tensor(clip_emb, dtype=torch.float32)       # (N, 768)
N = emb.shape[0]

# ── 3. Sound Source Remixer (attention blending) ─────────────────────────────
#
# The SS2A Remixer works by computing how much each detected sound source
# should contribute to the final blended audio representation.
# It uses scaled dot-product attention:
#   - Query and Key: the CLIP embeddings themselves
#   - Value: same embeddings (self-attention)
#   - Temperature 0.07: standard CLIP value, sharpens the weight distribution
#
# Result: each object's output embedding is a weighted mix of all objects,
# where objects with similar visual semantics attend strongly to each other.

temperature = 0.07

# Compute pairwise similarity between all N object embeddings
attn_logits  = torch.matmul(emb, emb.T)                 # (N, N)
attn_weights = F.softmax(attn_logits / temperature, dim=-1)  # (N, N)

# Weighted blend: each output = weighted sum of all input embeddings
remixed = torch.matmul(attn_weights, emb)               # (N, 768)

# Re-normalize to unit sphere (standard after blending)
remixed = F.normalize(remixed, dim=-1)                  # (N, 768)

# ── 4. Print attention summary ───────────────────────────────────────────────

print(f"\nAttention weight matrix : {attn_weights.shape}")
print("Top attending pairs (which objects attend most to each other):")

attn_np = attn_weights.detach().numpy()
for i in range(N):
    top_j = np.argsort(attn_np[i])[::-1][:3]           # top 3 for each object
    try:
        label_i = detections[i].get("label", f"obj_{i}")
        top_labels = [detections[j].get("label", f"obj_{j}") for j in top_j]
    except Exception:
        label_i = f"obj_{i}"
        top_labels = [f"obj_{j}" for j in top_j]
    print(f"  {label_i:20s} → attends to: {top_labels}")

# ── 5. Save output ───────────────────────────────────────────────────────────

out_path = "data/remixer_output.npy"
np.save(out_path, remixed.detach().numpy())

print(f"\nRemixer output shape : {remixed.shape}")
print(f"Saved                : {out_path}")

# ── 6. Verify saved file ─────────────────────────────────────────────────────

loaded = np.load(out_path)
print(f"Verified shape       : {loaded.shape}")
print(f"Value range          : min={loaded.min():.4f}, max={loaded.max():.4f}")
print("\nStage C complete. Ready for integration.")