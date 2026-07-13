import os
import glob
import random
import json
from collections import defaultdict
ROOT_DIR = 'c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\dataset\\full_dataset'
OUT_FILE = 'c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\dataset\\dataset_splits.json'

def main():
    random.seed(42)
    splits = {'train': [], 'val': [], 'test': []}
    total_images = 0
    collection_counts = {}
    for collection in os.listdir(ROOT_DIR):
        c_path = os.path.join(ROOT_DIR, collection)
        if not os.path.isdir(c_path):
            continue
        img_files = glob.glob(os.path.join(c_path, 'img', '*.*'))
        if not img_files:
            continue
        id_map = defaultdict(list)
        for f in img_files:
            basename = os.path.basename(f)
            file_id = os.path.splitext(basename)[0]
            rel_path = f'{collection}/img/{basename}'
            id_map[f'{collection}/{file_id}'].append(rel_path)
        unique_ids = list(id_map.keys())
        unique_ids.sort()
        random.shuffle(unique_ids)
        n = len(unique_ids)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_ids = unique_ids[:n_train]
        val_ids = unique_ids[n_train:n_train + n_val]
        test_ids = unique_ids[n_train + n_val:]
        if n > 0 and len(test_ids) == 0:
            test_ids = [train_ids.pop()]
        for tid in train_ids:
            splits['train'].extend(id_map[tid])
        for vid in val_ids:
            splits['val'].extend(id_map[vid])
        for tsid in test_ids:
            splits['test'].extend(id_map[tsid])
        collection_counts[collection] = len(img_files)
        total_images += len(img_files)
    with open(OUT_FILE, 'w') as f:
        json.dump(splits, f, indent=4)
    weights_map = {}
    for coll, count in collection_counts.items():
        weight = 1.0 / count if count > 0 else 0
        weights_map[coll] = weight
    weights_file = 'c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\dataset\\class_weights.json'
    with open(weights_file, 'w') as f:
        json.dump(weights_map, f, indent=4)
if __name__ == '__main__':
    main()