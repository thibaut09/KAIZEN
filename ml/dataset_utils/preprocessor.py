import os
import json
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
SPLITS_FILE = 'c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\dataset\\dataset_splits.json'
IN_ROOT = Path('c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\dataset\\full_dataset')
OUT_ROOT = Path('c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\dataset\\preprocessed')
PIXEL_ART = ['CRYPTOPUNKS', 'ToadPunks', '8liens']

def process_image(img_path, target_size=224, fill=128, is_pixel_art=False):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return (None, None)
    w, h = img.size
    max_wh = max(w, h)
    hp = (max_wh - w) // 2
    vp = (max_wh - h) // 2
    padding = (hp, vp, hp, vp)
    padded_img = F.pad(img, padding, fill, 'constant')
    interp = F.InterpolationMode.NEAREST if is_pixel_art else F.InterpolationMode.BICUBIC
    resized_img = F.resize(padded_img, [target_size, target_size], interpolation=interp)
    mask = Image.new('L', (w, h), 255)
    padded_mask = F.pad(mask, padding, 0, 'constant')
    resized_mask = F.resize(padded_mask, [target_size, target_size], interpolation=F.InterpolationMode.NEAREST)
    return (resized_img, resized_mask)

def main():
    if not os.path.exists(SPLITS_FILE):
        return
    with open(SPLITS_FILE, 'r') as f:
        splits = json.load(f)
    for split_name, file_paths in splits.items():
        for p in tqdm(file_paths, desc=f'{split_name.capitalize()}'):
            full_path = IN_ROOT / p
            if not full_path.exists():
                continue
            collection = p.split('/')[0]
            basename = os.path.basename(p)
            name = os.path.splitext(basename)[0]
            is_pixel = collection in PIXEL_ART
            img_out, mask_out = process_image(full_path, is_pixel_art=is_pixel)
            if img_out is None:
                continue
            out_dir = OUT_ROOT / split_name / collection
            out_dir.mkdir(parents=True, exist_ok=True)
            img_out.save(out_dir / f'{name}.png')
            mask_out.save(out_dir / f'{name}_mask.png')
if __name__ == '__main__':
    main()