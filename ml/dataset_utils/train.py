import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import roc_curve
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FEATURES_DIR = Path('c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\dataset\\extracted_features')
OUTPUT_WEIGHTS = 'c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\experiments\\final_weights\\experiment_4_thesis_vanilla.pth'

class projection_head(nn.Module):

    def __init__(self, input_dim=1536, hidden_dim=512, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

class fast_triplet_dataset(Dataset):

    def __init__(self, features_dict, valid_pristine, pristine_to_attacks, collection_to_pristine):
        self.features_dict = features_dict
        self.anchors = valid_pristine
        self.pristine_to_attacks = pristine_to_attacks
        self.collection_to_pristine = collection_to_pristine

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        anchor_key = self.anchors[idx]
        coll = anchor_key.split('/')[0]
        vec_a = self.features_dict[anchor_key]
        attacks = self.pristine_to_attacks[anchor_key]
        pos_key = random.choice(attacks)
        vec_p = self.features_dict[pos_key]
        siblings = self.collection_to_pristine[coll]
        siblings = [s for s in siblings if s != anchor_key]
        if not siblings:
            siblings = [s for s in self.anchors if s != anchor_key]
        neg_key = random.choice(siblings)
        vec_n = self.features_dict[neg_key]
        return (torch.tensor(vec_a), torch.tensor(vec_p), torch.tensor(vec_n))

def build_datasets():
    features_dict = {}
    for bf in sorted(FEATURES_DIR.glob('dual_features_train_batch_*.pt')):
        data = torch.load(bf, map_location='cpu', weights_only=False)['vanilla']
        features_dict.update(data)
    for bf in sorted(FEATURES_DIR.glob('dual_features_test_batch_*.pt')):
        data = torch.load(bf, map_location='cpu', weights_only=False)['vanilla']
        features_dict.update(data)
    pristine_keys = [k for k in features_dict.keys() if '_attack_' not in k]
    attack_keys = [k for k in features_dict.keys() if '_attack_' in k]
    pristine_to_attacks = {k: [] for k in pristine_keys}
    for ak in attack_keys:
        parts = ak.split('_attack_')
        anchor_k = f'{parts[0]}.png' if not parts[0].endswith('.png') else parts[0]
        if anchor_k in pristine_to_attacks:
            pristine_to_attacks[anchor_k].append(ak)
    valid_pristine = [k for k, v in pristine_to_attacks.items() if len(v) > 0]
    collection_to_pristine = {}
    for k in valid_pristine:
        coll = k.split('/')[0]
        if coll not in collection_to_pristine:
            collection_to_pristine[coll] = []
        collection_to_pristine[coll].append(k)
    collections = list(collection_to_pristine.keys())
    random.seed(42)
    random.shuffle(collections)
    split_idx = int(len(collections) * 0.8)
    train_colls = set(collections[:split_idx])
    val_colls = set(collections[split_idx:])
    train_anchors = [k for k in valid_pristine if k.split('/')[0] in train_colls]
    val_anchors = [k for k in valid_pristine if k.split('/')[0] in val_colls]
    train_ds = fast_triplet_dataset(features_dict, train_anchors, pristine_to_attacks, collection_to_pristine)
    val_ds = fast_triplet_dataset(features_dict, val_anchors, pristine_to_attacks, collection_to_pristine)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    return (train_loader, val_loader)

def train_model():
    os.makedirs(os.path.dirname(OUTPUT_WEIGHTS), exist_ok=True)
    model = projection_head().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.TripletMarginLoss(margin=0.5, p=2.0)
    train_loader, val_loader = build_datasets()
    epochs = 15
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for vec_a, vec_p, vec_n in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False):
            vec_a = vec_a.to(DEVICE, non_blocking=True)
            vec_p = vec_p.to(DEVICE, non_blocking=True)
            vec_n = vec_n.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            proj_a = model(vec_a)
            proj_p = model(vec_p)
            proj_n = model(vec_n)
            loss = criterion(proj_a, proj_p, proj_n)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0.0
        all_sims = []
        all_targets = []
        with torch.no_grad():
            for vec_a, vec_p, vec_n in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]', leave=False):
                vec_a = vec_a.to(DEVICE, non_blocking=True)
                vec_p = vec_p.to(DEVICE, non_blocking=True)
                vec_n = vec_n.to(DEVICE, non_blocking=True)
                proj_a = model(vec_a)
                proj_p = model(vec_p)
                proj_n = model(vec_n)
                loss = criterion(proj_a, proj_p, proj_n)
                val_loss += loss.item()
                pos_sim = F.cosine_similarity(proj_a, proj_p).cpu().numpy()
                neg_sim = F.cosine_similarity(proj_a, proj_n).cpu().numpy()
                all_sims.extend(pos_sim)
                all_targets.extend(np.ones_like(pos_sim))
                all_sims.extend(neg_sim)
                all_targets.extend(np.zeros_like(neg_sim))
        val_loss /= len(val_loader)
        fpr, tpr, _ = roc_curve(all_targets, all_sims)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.absolute(fnr - fpr))
        val_eer = fpr[eer_idx]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_WEIGHTS)
        else:
            patience_counter += 1
            if patience_counter >= 3:
                break
if __name__ == '__main__':
    train_model()