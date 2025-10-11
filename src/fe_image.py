import os, io
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

def build_resnet50():
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    m.fc = nn.Identity()  # 2048-d
    m.eval()
    return m

def img_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

@torch.no_grad()
def embed_images(df, image_folder='assets/images', batch_size=32, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_resnet50().to(device)
    tfm = img_transform()

    feats = np.zeros((len(df), 2048), dtype='float32')
    miss = np.zeros((len(df),), dtype='int8')
    idx_map = []
    batch_imgs = []
    batch_idx = []

    def flush():
        nonlocal batch_imgs, batch_idx
        if not batch_imgs: return
        x = torch.stack(batch_imgs).to(device)
        f = model(x).cpu().numpy()
        feats[np.array(batch_idx)] = f
        batch_imgs, batch_idx = [], []

    for i, sid in enumerate(tqdm(df['sample_id'], desc='embed')):
        path = os.path.join(image_folder, f"{sid}.jpg")
        if not os.path.exists(path):
            miss[i] = 1
            idx_map.append(i)
            continue
        try:
            im = Image.open(path).convert('RGB')
            x = tfm(im)
            batch_imgs.append(x)
            batch_idx.append(i)
            if len(batch_imgs) >= batch_size:
                flush()
        except:
            miss[i] = 1
    flush()
    return feats, miss

def save_image_features(train_df, test_df, out_prefix='features/img'):
    # returns and saves train/test embeddings + missing flags
    tr_feats, tr_miss = embed_images(train_df)
    te_feats, te_miss = embed_images(test_df)
    np.save(f'{out_prefix}_train.npy', tr_feats)
    np.save(f'{out_prefix}_test.npy',  te_feats)
    np.save(f'{out_prefix}_train_missing.npy', tr_miss)
    np.save(f'{out_prefix}_test_missing.npy',  te_miss)
