import os
import torch
import random
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path

class dual_vector_dataset(Dataset):
    def __init__(self, features_dir, split="train", use_masked=True, use_hybrid=False, weights_file=None):
        self.features_dir = Path(features_dir)
        self.split = split
        self.use_masked = use_masked
        self.use_hybrid = use_hybrid
        batch_files = list(self.features_dir.glob(f"dual_features_{split}_batch_*.pt"))
        if not batch_files:
            raise FileNotFoundError(f"No feature batches found for split '{split}' in {self.features_dir}")
            
        self.pristine_dict = {}
        self.attack_dict = {}
        
        print(f"[Dataset] Loading features for {split.upper()} split (Hybrid: {self.use_hybrid}, Masked: {self.use_masked})...")
        batch_files = sorted(batch_files)
        for bf in batch_files:
            data = torch.load(bf, map_location="cpu", weights_only=False)
            
            if self.use_hybrid:
                v_feats = data.get("vanilla", {})
                m_feats = data.get("masked", {})
                for k in v_feats.keys():
                    if k in m_feats:
                        combined = np.concatenate([v_feats[k], m_feats[k]], axis=0)
                        if "_attack_" in k:
                            self.attack_dict[k] = combined
                        else:
                            self.pristine_dict[k] = combined
            else:
                feature_key = "masked" if use_masked else "vanilla"
                features = data.get(feature_key, {})
                for k, v in features.items():
                    if "_attack_" in k:
                        self.attack_dict[k] = v
                    else:
                        self.pristine_dict[k] = v
                    
        attack_prefixes = set()
        for ak in self.attack_dict.keys():
            attack_prefixes.add(ak.split('_attack_')[0])
            
        valid_pristine = []
        for k in self.pristine_dict.keys():
            coll = k.split('/')[0]
            basename = Path(k).stem
            if f"{coll}/{basename}" in attack_prefixes:
                valid_pristine.append(k)
                
        self.pristine_keys = valid_pristine
        print(f"[Dataset] Filtered to {len(self.pristine_keys)} strictly valid images (having true attacks).")
        
        self.pristine_to_attacks = {}
        for k in self.pristine_keys:
            coll = k.split('/')[0]
            basename = Path(k).stem
            self.pristine_to_attacks[k] = [ak for ak in self.attack_dict.keys() if ak.startswith(f"{coll}/{basename}_attack_")]
        
        self.collection_to_pristine = {}
        for k in self.pristine_keys:
            coll = k.split('/')[0]
            if coll not in self.collection_to_pristine:
                self.collection_to_pristine[coll] = []
            self.collection_to_pristine[coll].append(k)
            
        self.weights = []
        if weights_file and os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                class_weights = json.load(f)
            for k in self.pristine_keys:
                coll = k.split('/')[0]
                self.weights.append(class_weights.get(coll, 1.0))

    def __len__(self):
        return len(self.pristine_keys)

    def __getitem__(self, idx):
        anchor_key = self.pristine_keys[idx]
        anchor_vec = self.pristine_dict[anchor_key]
        coll = anchor_key.split('/')[0]
        basename = Path(anchor_key).stem
        
        if random.random() > 0.5:
            possible_attacks = self.pristine_to_attacks[anchor_key]
            attack_key = random.choice(possible_attacks)
            return anchor_vec, self.attack_dict[attack_key], 1.0
        else:
            if random.random() < 0.7:
                siblings = [k for k in self.collection_to_pristine[coll] if k != anchor_key]
                if siblings:
                    sibling_key = random.choice(siblings)
                    return anchor_vec, self.pristine_dict[sibling_key], -1.0
            
            other_colls = [c for c in self.collection_to_pristine.keys() if c != coll]
            if other_colls:
                neg_coll = random.choice(other_colls)
                neg_key = random.choice(self.collection_to_pristine[neg_coll])
                return anchor_vec, self.pristine_dict[neg_key], -1.0
            else:
                return anchor_vec, anchor_vec, -1.0

def get_dataloader(features_dir, split="train", use_masked=True, use_hybrid=False, weights_file=None, batch_size=512):
    
    dataset = dual_vector_dataset(features_dir, split, use_masked, use_hybrid, weights_file)
    
    sampler = None
    if split == "train" and len(dataset.weights) > 0:
        sampler = WeightedRandomSampler(weights=dataset.weights, num_samples=len(dataset), replacement=True)
        
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        shuffle=(sampler is None and split == "train"),
        num_workers=0,
        pin_memory=True
    )
    return loader
