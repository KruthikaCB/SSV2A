# explore_clip.py
# Kavana - CLIP Embedding Module
# Passes each crop through CLIP ViT-L/14, saves 768-dim embeddings

import json, numpy as np, torch, clip
from PIL import Image
from pathlib import Path

def embed_crops(detections_path='data/detections.json',
                 output_npy='data/clip_embeddings.npy',
                 output_labels='data/clip_labels.json'):

    with open(detections_path) as f:
        detections = json.load(f)
    print(f'Loaded {len(detections)} detections')

    print('Loading CLIP ViT-L/14 (first run downloads ~890MB)...')
    model, preprocess = clip.load('ViT-L/14', device='cpu')
    model.eval()
    print('CLIP loaded')

    embeddings = []
    labels = []

    with torch.no_grad():
        for i, det in enumerate(detections):
            crop = Image.open(det['crop_file']).convert('RGB')
            inp = preprocess(crop).unsqueeze(0)
            emb = model.encode_image(inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy()[0])
            labels.append({
                'index': i,
                'label': det['label'],
                'confidence': det['confidence'],
                'crop_file': det['crop_file'],
                'embedding_row': len(embeddings) - 1
            })
            print(f'  Embedded {det["label"]:20s} shape: (768,)')

    matrix = np.stack(embeddings)
    np.save(output_npy, matrix)

    with open(output_labels, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f'Embeddings shape: {matrix.shape}')
    print(f'Saved: {output_npy} + {output_labels}')
    return matrix, labels

if __name__ == '__main__':
    embed_crops()