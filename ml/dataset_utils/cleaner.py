import os
import glob
from PIL import Image
import imagehash
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import pdist, squareform
ROOT_DIR = 'c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\dataset\\full_dataset'

def deduplicate_collection(collection_path):
    img_files = glob.glob(os.path.join(collection_path, 'img', '*.*'))
    if not img_files:
        return 0
    hashes = []
    sizes = []
    valid_files = []
    for f in tqdm(img_files, desc='Hashing', leave=False):
        try:
            with Image.open(f) as img:
                h = imagehash.phash(img)
                hashes.append(h.hash.flatten())
                sizes.append(img.size[0] * img.size[1])
                valid_files.append(f)
        except Exception as e:
    if len(valid_files) < 2:
        return 0
    THRESHOLD = 3
    hashes_arr = np.array(hashes, dtype=bool)
    sizes_arr = np.array(sizes)
    dists = pdist(hashes_arr, metric='hamming') * 64
    dist_matrix = squareform(dists)
    np.fill_diagonal(dist_matrix, 999)
    pairs = np.argwhere(dist_matrix <= THRESHOLD)
    to_delete = set()
    for i, j in pairs:
        if i in to_delete or j in to_delete:
            continue
        if sizes_arr[i] >= sizes_arr[j]:
            to_delete.add(j)
        else:
            to_delete.add(i)
    deleted_count = 0
    for idx in to_delete:
        os.remove(valid_files[idx])
        deleted_count += 1
    return deleted_count

def main():
    os.makedirs('c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\ml\\dataset_utils', exist_ok=True)
    total_deleted = 0
    for collection in os.listdir(ROOT_DIR):
        c_path = os.path.join(ROOT_DIR, collection)
        if os.path.isdir(c_path):
            count = deduplicate_collection(c_path)
            total_deleted += count
            if count > 0:
if __name__ == '__main__':
    main()