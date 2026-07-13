import os
import json
import faiss
import numpy as np
from tqdm import tqdm

def main():
    prod_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'prod_index_exp6'))
    idx_dino = faiss.read_index(os.path.join(prod_dir, 'dino_index.faiss'))
    idx_siglip = faiss.read_index(os.path.join(prod_dir, 'siglip_index.faiss'))
    idx_proj = faiss.read_index(os.path.join(prod_dir, 'proj_index.faiss'))
    with open(os.path.join(prod_dir, 'metadata_map.json'), 'r') as f:
        metadata = json.load(f)
    n_images = len(metadata)
    raw_vecs = []
    proj_db = {}
    for i in tqdm(range(n_images)):
        d_v = idx_dino.reconstruct(i)
        s_v = idx_siglip.reconstruct(i)
        p_v = idx_proj.reconstruct(i)
        d_v = d_v / np.linalg.norm(d_v)
        s_v = s_v / np.linalg.norm(s_v)
        f_v = np.concatenate([d_v, s_v])
        raw_vecs.append(f_v)
        img_hash = metadata[i]['hash']
        proj_db[img_hash] = p_v.tolist()
    idx_raw = faiss.IndexFlatIP(1536)
    idx_raw.add(np.stack(raw_vecs).astype(np.float32))
    faiss.write_index(idx_raw, os.path.join(prod_dir, 'raw_index.faiss'))
    with open(os.path.join(prod_dir, 'proj_db.json'), 'w') as f:
        json.dump(proj_db, f)
if __name__ == '__main__':
    main()