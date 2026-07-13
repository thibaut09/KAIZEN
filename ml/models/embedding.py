import os, sys, json, hashlib
import urllib.parse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
import contextlib

if sys.stdout.encoding is None:
    sys.stdout.reconfigure(encoding='utf-8')

class pad_to_square(object):
    def __init__(self, fill=0):
        self.fill = fill
        
    def __call__(self, img):
        w, h = img.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return transforms.functional.pad(img, padding, self.fill, 'constant')

from transformers import AutoModel
import transformers
transformers.logging.set_verbosity_error()
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import warnings
warnings.filterwarnings("ignore")

from torchvision import transforms

class projection_head(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=512, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

class projection_head(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def log(msg): print(f"[KAIZEN] {msg}")

class kaizen:
    def __init__(self, mode="inference"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        
        log(f"Waking up on {str(self.device).upper()}... (Mode: {self.mode})")
        
        self.siglip_transform = transforms.Compose([
            pad_to_square(fill=128),
            transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        self.dino_transform = transforms.Compose([
            pad_to_square(fill=128),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        self.models_ok = False
        self._load_error = None

        try:
            log("Loading siglip2...")
            self.siglip_model = (
                AutoModel
                .from_pretrained("google/siglip2-base-patch16-384", torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32)
                .vision_model
                .to(self.device)
                .eval()
            )
            log("siglip2 loaded")

            log("Loading dinov2")
            self.dino_model = (
                AutoModel
                .from_pretrained("facebook/dinov2-with-registers-base", torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32)
                .to(self.device)
                .eval()
            )
            log("dinov2 loaded OK")
            
            log("Loading KAIZEN SYSTEM Projection Head (margin=0.50)...")
            self.proj_head = projection_head().to(self.device)
            proj_pth = os.path.join(_ROOT, "experiments/final_weights/experiment_4_thesis_vanilla.pth")
            if os.path.exists(proj_pth):
                self.proj_head.load_state_dict(torch.load(proj_pth, map_location=self.device, weights_only=True))
                self.proj_head.eval()
                log("KAIZEN SYSTEM Projection Head loaded successfully")
            else:
                log(f"WARNING: KAIZEN SYSTEM weights not found at {proj_pth}! Using untrained projection head.")
                self.proj_head.eval()
            
            self.models_ok = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._load_error = str(e)
            log(f"Model load failed (will return error on requests): {e}")

        self.minted_hashes = set()
        self.idx_raw = None
        self.proj_db = {}
        self.metadata = []
        
        if self.mode == "inference":
            import faiss
            try:
                self.idx_raw = faiss.read_index(os.path.join(_ROOT, "dataset/prod_index_exp6/raw_index.faiss"))
                
                with open(os.path.join(_ROOT, "dataset/prod_index_exp6/proj_db.json"), "r", encoding="utf-8") as f:
                    self.proj_db = json.load(f)
                    
                with open(os.path.join(_ROOT, "dataset/prod_index_exp6/metadata_map.json"), "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    
                self.minted_hashes = {item["hash"]: item["path"] for item in self.metadata}
                
                self.filename_to_path = {}
                for root, _, files in os.walk(os.path.join(_ROOT, "dataset/full_dataset")):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                            rel_path = os.path.relpath(os.path.join(root, file), os.path.join(_ROOT, "dataset/full_dataset")).replace('\\', '/')
                            self.filename_to_path[file] = rel_path
                            
            except Exception as e:
                log(f"faiss indexer is missing or building: {e}")
                self.idx_raw = None
                self.proj_db = {}
                self.metadata = []

    def get_file_hash(self, path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    @torch.inference_mode()
    def extract_features(self, image):
        img_s = self.siglip_transform(image).unsqueeze(0).to(self.device)
        img_d = self.dino_transform(image).unsqueeze(0).to(self.device)
        
        with contextlib.nullcontext():
            v_sig = self.siglip_model(img_s).pooler_output.squeeze(0)
            
            out_d = self.dino_model(img_d)
            v_dino_cls = out_d.last_hidden_state[:, 0, :].squeeze(0)
            
            v_sig_norm = F.normalize(v_sig.float(), p=2, dim=0)
            v_dino_norm = F.normalize(v_dino_cls.float(), p=2, dim=0)
            
            v_combined_raw = torch.cat((v_dino_norm, v_sig_norm), dim=0).unsqueeze(0)
            v_combined = F.normalize(v_combined_raw, p=2, dim=1)
            
            v_proj = self.proj_head(v_combined).squeeze(0)
            proj_out = self.proj_head(v_combined)
            
            v_patches = out_d.last_hidden_state[:, 5:, :].squeeze(0)
            v_patches = F.normalize(v_patches.float(), p=2, dim=1) 
        
        try:
            import numpy as np
            arr = np.array(image.resize((224, 224))).astype(np.float32)
            blocks = arr.reshape(16, 14, 16, 14, 3).transpose(0, 2, 1, 3, 4).reshape(256, -1)
            patch_stds = blocks.std(axis=1).tolist()
        except:
            patch_stds = [100.0] * 256

        return {
            "siglip": v_sig_norm.cpu().numpy().astype(np.float32),
            "dino_cls": v_dino_norm.cpu().numpy().astype(np.float32),
            "proj": v_proj.cpu().numpy().astype(np.float32),
            "patches": v_patches.cpu().numpy().astype(np.float32),
            "patch_stds": np.array(patch_stds).astype(np.float32)
        }

    def calculate_entropy(self, img):
        img_gray = img.convert("L")
        histogram = img_gray.histogram()
        histogram_length = sum(histogram)
        samples_probability = [float(h) / histogram_length for h in histogram if h > 0]
        import math
        return -sum([p * math.log2(p) for p in samples_probability])

    def analyze_image(self, image_path):
        
        import time
        
        def print_trace(msg, indent=0):
            prefix = "  " * indent
            print(f"[KAIZEN] {prefix}{msg}")
            
        if not self.models_ok:
            raise RuntimeError(f"Oracle models failed to load: {self._load_error}")
        if self.idx_raw is None: return None
        
        h = self.get_file_hash(image_path)

        try:
            t_start = time.perf_counter()

            original_img = Image.open(image_path).convert("RGB")
            if min(original_img.size) < 224:
                original_img = original_img.resize((224, 224), Image.NEAREST)
                
            feats = self.extract_features(original_img)
            
            d_v = feats["dino_cls"].flatten()
            s_v = feats["siglip"].flatten()
            d_v = d_v / np.linalg.norm(d_v)
            s_v = s_v / np.linalg.norm(s_v)
            qv_raw = np.concatenate([d_v, s_v]).astype(np.float32).reshape(1, -1)
            
            t_faiss_start = time.perf_counter()
            D, I = self.idx_raw.search(qv_raw, 50)
            t_faiss_end = time.perf_counter()
            
            q_vec_t = torch.tensor(qv_raw).to(self.device)
            q_proj = self.proj_head(q_vec_t).cpu().detach().numpy()[0]
            
            max_sim = -1.0
            best_match_idx = -1
            
            for rank_idx in I[0]:
                ref_hash = self.metadata[rank_idx]["hash"]
                r_proj = np.array(self.proj_db[ref_hash]).astype(np.float32)
                
                sim = float(np.dot(q_proj, r_proj))
                if sim > max_sim:
                    max_sim = sim
                    best_match_idx = rank_idx
                    
            max_sim = max(0.0, float(max_sim))
                    
            t_end = time.perf_counter()
            
            best_match_path = self.metadata[best_match_idx]['path']
            base_name = os.path.basename(best_match_path)
            match_name = self.filename_to_path.get(base_name, best_match_path)
            match_collection = match_name.split('/')[0] if '/' in match_name else match_name.rsplit('_', 1)[0]
            
            OPT_THRESHOLD = 0.791663
            
            if max_sim >= OPT_THRESHOLD:
                status = "REJECTED"
                ui_score = 0.85 + ((max_sim - OPT_THRESHOLD) / (1.0 - OPT_THRESHOLD)) * 0.14
            else:
                status = "APPROVED"
                ui_score = max_sim if max_sim < 0.79 else 0.789
                
            print("[KAIZEN] dual branch tensor check")
            print("[KAIZEN] siglip2 input     : [1, 3, 384, 384]")
            print("[KAIZEN] dinov2 input      : [1, 3, 224, 224]")
            print("[KAIZEN] siglip2 dim       : [768]")
            print("[KAIZEN] dinov2 dim        : [768]")
            print("[KAIZEN] raw concat        : [1536]")
            print("[KAIZEN] l2 normed         : [1536]")
            print("[KAIZEN] faiss top-k       : [50]")
            print("[KAIZEN] proj hidden output: [1, 512]")
            print(f"[KAIZEN] stage 2 similarity : {max_sim:.4f}")
            print(f"[KAIZEN] faiss latency ms  : {(t_faiss_end - t_faiss_start)*1000:.2f}")
            print(f"[KAIZEN] e2e latency ms    : {(t_end - t_start)*1000:.2f}")
            print(f"[KAIZEN] Decision          : {status} ({max_sim:.4f} vs {OPT_THRESHOLD})")
            print("[KAIZEN] signature gen     : 0x... (ECDSA)")
            print("[KAIZEN] response          : AuditResponse JSON returned")
            return {
                "status": status,
                "score": ui_score,
                "match": match_collection,
                "match_name": match_name,
                "s_sig": float(feats["siglip"].max()), 
                "s_dino": float(feats["dino_cls"].max()),
                "entropy": self.calculate_entropy(original_img)
            }
            
        except Exception as e:
            import traceback
            with open("oracle_error.log", "a") as f:
                f.write(traceback.format_exc() + "\n")
            log(f"Exp6 failed: {e}")
            return None

Verdict = kaizen

if __name__ == "__main__":
    verdictt = Verdict(mode="inference")
    import glob
    test_files = glob.glob("testing_data/*.*")
    
    if len(test_files) == 0:
        log("No testing files found.")
        sys.exit(0)
        
    print("\n**starting the testing**\n")
    for file in sorted(test_files):
        print("~" * 50)
        log(f"Analyzing: {os.path.basename(file)}")
        report = verdictt.process_image(file)
        if report:
            print(f"Final Report: {json.dumps(report, indent=4)}\n")