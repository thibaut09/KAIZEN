import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel
import math
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 500
INFER_BATCH_SIZE = 32
IN_ROOT = Path('c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\dataset\\preprocessed')
OUT_ROOT = Path('c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\dataset\\extracted_features')

class minimal_processor:

    def __init__(self, target_size, mean, std):
        self.transform = transforms.Compose([transforms.Resize((target_size, target_size)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    def __call__(self, img):
        return self.transform(img)
dino_proc = minimal_processor(224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
siglip_proc = minimal_processor(384, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

class fast_extract_dataset(Dataset):
    
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        maskpath = filepath.parent / f'{filepath.stem}_mask.png'
        if not maskpath.exists():
            return None
        try:
            img = Image.open(filepath).convert('RGB')
            mask_img = Image.open(maskpath).convert('L')
            d_tensor = dino_proc(img)
            d_mask = transforms.functional.resize(mask_img, (224, 224), transforms.InterpolationMode.NEAREST)
            d_mask = transforms.functional.to_tensor(d_mask)
            s_tensor = siglip_proc(img)
            s_mask = transforms.functional.resize(mask_img, (384, 384), transforms.InterpolationMode.NEAREST)
            s_mask = transforms.functional.to_tensor(s_mask)
            collection_name = filepath.parent.name
            key = f'{collection_name}/{filepath.name}'
            return (d_tensor, d_mask, s_tensor, s_mask, key)
        except Exception as e:
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    d_tensors, d_masks, s_tensors, s_masks, keys = zip(*batch)
    return (torch.stack(d_tensors), torch.stack(d_masks), torch.stack(s_tensors), torch.stack(s_masks), keys)

def extract_features():
    os.makedirs(OUT_ROOT, exist_ok=True)
    dinov2_model = AutoModel.from_pretrained('facebook/dinov2-with-registers-base').to(DEVICE).eval()
    siglip2_model = AutoModel.from_pretrained('google/siglip2-base-patch16-384').to(DEVICE).eval()
    splits = ['val', 'test']
    num_workers = min(4, os.cpu_count() or 1)
    for split in splits:
        split_dir = IN_ROOT / split
        if not split_dir.exists():
            continue
        all_files = [f for f in split_dir.rglob('*.png') if not f.name.endswith('_mask.png')]
        total_batches = math.ceil(len(all_files) / BATCH_SIZE)
        for batch_idx in range(total_batches):
            out_path = OUT_ROOT / f'dual_features_{split}_batch_{batch_idx}.pt'
            if out_path.exists():
                continue
            batch_files = all_files[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
            dataset = fast_extract_dataset(batch_files)
            dataloader = DataLoader(dataset, batch_size=INFER_BATCH_SIZE, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
            batch_data = {'vanilla': {}, 'masked': {}}
            for batch_res in tqdm(dataloader, desc=f'Batch {batch_idx + 1}/{total_batches}'):
                if batch_res is None:
                    continue
                d_batch, d_mask_batch, s_batch, s_mask_batch, keys = batch_res
                d_batch = d_batch.to(DEVICE, non_blocking=True)
                d_mask_batch = d_mask_batch.to(DEVICE, non_blocking=True)
                s_batch = s_batch.to(DEVICE, non_blocking=True)
                s_mask_batch = s_mask_batch.to(DEVICE, non_blocking=True)
                d_patch_mask = F.avg_pool2d(d_mask_batch, kernel_size=14, stride=14)
                d_patch_mask = (d_patch_mask > 0.5).float().view(d_batch.shape[0], -1, 1)
                s_patch_mask = F.avg_pool2d(s_mask_batch, kernel_size=16, stride=16)
                s_patch_mask = (s_patch_mask > 0.5).float().view(s_batch.shape[0], -1, 1)
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        d_out = dinov2_model(d_batch, output_hidden_states=True)
                        d_layer9 = d_out.hidden_states[9]
                        d_cls = F.normalize(d_out.last_hidden_state[:, 0, :], p=2, dim=-1)
                        d_spatial = d_layer9[:, 5:, :] * d_patch_mask
                        d_active_count = torch.clamp(d_patch_mask.sum(dim=1), min=1e-09)
                        d_masked_pool = F.normalize(d_spatial.sum(dim=1) / d_active_count, p=2, dim=-1)
                        s_out = siglip2_model.vision_model(s_batch)
                        s_pooler = F.normalize(s_out.pooler_output, p=2, dim=-1)
                        s_spatial = s_out.last_hidden_state * s_patch_mask
                        s_active_count = torch.clamp(s_patch_mask.sum(dim=1), min=1e-09)
                        s_masked_pool = F.normalize(s_spatial.sum(dim=1) / s_active_count, p=2, dim=-1)
                vanilla_vec = F.normalize(torch.cat((d_cls, s_pooler), dim=-1), p=2, dim=-1).cpu().numpy().astype(np.float32)
                masked_vec = F.normalize(torch.cat((d_masked_pool, s_masked_pool), dim=-1), p=2, dim=-1).cpu().numpy().astype(np.float32)
                for idx, k in enumerate(keys):
                    batch_data['vanilla'][k] = vanilla_vec[idx]
                    batch_data['masked'][k] = masked_vec[idx]
            torch.save(batch_data, out_path)
if __name__ == '__main__':
    extract_features()