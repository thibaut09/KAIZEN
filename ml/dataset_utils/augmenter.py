import os
import glob
import random
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm
from pathlib import Path
TRAIN_DIR = Path('c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\dataset\\preprocessed\\train')

def generate_hard_negatives(img_path, mask_path):
    try:
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
    except:
        return []
    out_dir = img_path.parent
    basename = img_path.stem
    generated = []
    angle = random.uniform(-60, 60)
    translate = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))
    scale = random.uniform(0.5, 1.5)
    shear = random.uniform(-30, 30)
    img_affine = F.affine(img, angle=angle, translate=[int(translate[0] * 224), int(translate[1] * 224)], scale=scale, shear=[shear], fill=128)
    mask_affine = F.affine(mask, angle=angle, translate=[int(translate[0] * 224), int(translate[1] * 224)], scale=scale, shear=[shear], fill=0)
    affine_path = out_dir / f'{basename}_attack_affine.png'
    affine_mask_path = out_dir / f'{basename}_attack_affine_mask.png'
    img_affine.save(affine_path)
    mask_affine.save(affine_mask_path)
    generated.append(affine_path)
    tensor_img = F.to_tensor(img)
    tensor_mask = F.to_tensor(mask)
    h, w = (tensor_img.shape[1], tensor_img.shape[2])
    area = h * w
    erase_area = random.uniform(0.4, 0.7) * area
    aspect_ratio = random.uniform(0.3, 3.3)
    h_erase = int(round((erase_area * aspect_ratio) ** 0.5))
    w_erase = int(round((erase_area / aspect_ratio) ** 0.5))
    if h_erase <= h and w_erase <= w:
        i = random.randint(0, h - h_erase)
        j = random.randint(0, w - w_erase)
        img_occ = tensor_img.clone()
        mask_occ = tensor_mask.clone()
        img_occ[:, i:i + h_erase, j:j + w_erase] = random.uniform(0, 1)
        mask_occ[:, i:i + h_erase, j:j + w_erase] = 0.0
        occ_path = out_dir / f'{basename}_attack_occlusion.png'
        occ_mask_path = out_dir / f'{basename}_attack_occlusion_mask.png'
        F.to_pil_image(img_occ).save(occ_path)
        F.to_pil_image(mask_occ).save(occ_mask_path)
        generated.append(occ_path)
    jitter = T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.4)
    img_diff = jitter(img)
    img_diff = F.gaussian_blur(img_diff, kernel_size=[9, 9], sigma=[3.0, 5.0])
    import numpy as np
    arr = np.array(img_diff).astype(np.float32)
    noise = np.random.normal(0, 50, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img_diff = Image.fromarray(arr)
    if random.random() < 0.2:
        from PIL import ImageOps
        img_diff = ImageOps.invert(img_diff)
    diff_path = out_dir / f'{basename}_attack_diffusion.png'
    diff_mask_path = out_dir / f'{basename}_attack_diffusion_mask.png'
    img_diff.save(diff_path)
    mask.save(diff_mask_path)
    generated.append(diff_path)
    return generated

def main():
    splits = ['val', 'test']
    base_dir = Path('c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\dataset\\preprocessed')
    total_attacks = 0
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
        collections = [d for d in os.listdir(split_dir) if (split_dir / d).is_dir()]
        for coll in collections:
            coll_dir = split_dir / coll
            pristine_files = [f for f in glob.glob(os.path.join(coll_dir, '*.png')) if not f.endswith('_mask.png') and '_attack_' not in f]
            for pf in tqdm(pristine_files, leave=False):
                img_path = Path(pf)
                mask_path = coll_dir / f'{img_path.stem}_mask.png'
                if not mask_path.exists():
                    continue
                generated = generate_hard_negatives(img_path, mask_path)
                total_attacks += len(generated)
if __name__ == '__main__':
    main()