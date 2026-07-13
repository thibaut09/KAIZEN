import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
from tqdm import tqdm
from pathlib import Path
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FEATURES_DIR = Path('c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\dataset\\extracted_features')
WEIGHTS_FILE = 'c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\experiments\\final_weights\\experiment_4_thesis_vanilla.pth'
HISTORY_ROOT = Path('c:\\Users\\lolip\\Downloads\\NFT Originality Detection\\experiments\\history_strict_jury\\Experiment_6_Asymmetric_Reranking')
os.makedirs(HISTORY_ROOT, exist_ok=True)
OPTIMAL_THRESHOLD = 0.791663
TOP_K_RECALL = 50

class projection_head(nn.Module):

    def __init__(self, input_dim=1536, hidden_dim=512, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

def run_asymmetric_pipeline():
    proj_head = projection_head().to(DEVICE)
    proj_head.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE, weights_only=False))
    proj_head.eval()
    batch_files = list(FEATURES_DIR.glob(f'dual_features_test_batch_*.pt'))
    pristine_dict = {}
    attack_dict = {}
    for bf in sorted(batch_files):
        data = torch.load(bf, map_location='cpu', weights_only=False)['vanilla']
        for k, v in data.items():
            if '_attack_' in k:
                attack_dict[k] = v
            else:
                pristine_dict[k] = v
    attack_prefixes = set([ak.split('_attack_')[0] for ak in attack_dict.keys()])
    valid_pristine = [k for k in pristine_dict.keys() if f"{k.split('/')[0]}/{Path(k).stem}" in attack_prefixes]
    idx_raw = faiss.IndexFlatIP(1536)
    ref_keys = []
    raw_vecs = []
    proj_db = {}
    with torch.no_grad():
        for i in range(0, len(valid_pristine), 256):
            batch_keys = valid_pristine[i:i + 256]
            batch_vecs = [pristine_dict[k] for k in batch_keys]
            for j, vec in enumerate(batch_vecs):
                d_v = vec[:768]
                s_v = vec[768:]
                d_v = d_v / np.linalg.norm(d_v)
                s_v = s_v / np.linalg.norm(s_v)
                f_v = np.concatenate([d_v, s_v])
                raw_vecs.append(f_v)
                ref_keys.append(batch_keys[j])
            t = torch.tensor(np.array(batch_vecs)).to(DEVICE)
            p = proj_head(t).cpu().numpy()
            for j, k in enumerate(batch_keys):
                proj_db[k] = p[j]
    idx_raw.add(np.stack(raw_vecs).astype(np.float32))
    queries = []
    for anchor_key in valid_pristine:
        coll = anchor_key.split('/')[0]
        basename = Path(anchor_key).stem
        attacks = [k for k in attack_dict.keys() if k.startswith(f'{coll}/{basename}_attack_')]
        if attacks:
            queries.append({'q_key': attacks[0], 'vec': attack_dict[attacks[0]], 'is_plagiarism': True, 'true_ref': anchor_key})
        siblings = [k for k in valid_pristine if k != anchor_key and k.startswith(f'{coll}/')]
        if siblings:
            queries.append({'q_key': siblings[0], 'vec': pristine_dict[siblings[0]], 'is_plagiarism': False, 'true_ref': siblings[0]})
    stats = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    end_to_end_latencies = []
    all_sims = []
    all_targets = []
    with torch.no_grad():
        for q in tqdm(queries, desc='Asymmetric Pipeline'):
            t_start = time.perf_counter()
            v = q['vec']
            d_v = v[:768]
            s_v = v[768:]
            d_v = d_v / np.linalg.norm(d_v)
            s_v = s_v / np.linalg.norm(s_v)
            qv_raw = np.concatenate([d_v, s_v]).astype(np.float32).reshape(1, -1)
            D, I = idx_raw.search(qv_raw, TOP_K_RECALL + 1)
            q_vec_t = torch.tensor(v).unsqueeze(0).to(DEVICE)
            q_proj = proj_head(q_vec_t).cpu().numpy()[0]
            max_sim = -1.0
            q_collection = q['q_key'].split('/')[0]
            for rank_idx in I[0]:
                ref_key = ref_keys[rank_idx]
                r_collection = ref_key.split('/')[0]
                if ref_key == q['q_key']:
                    continue
                if not q['is_plagiarism'] and q_collection == r_collection:
                    continue
                r_proj = proj_db[ref_key]
                sim = float(np.dot(q_proj, r_proj))
                if sim > max_sim:
                    max_sim = sim
            t_end = time.perf_counter()
            end_to_end_latencies.append((t_end - t_start) * 1000)
            all_sims.append(max_sim)
            all_targets.append(1 if q['is_plagiarism'] else 0)
    from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    all_targets = np.array(all_targets)
    all_sims = np.array(all_sims)
    fpr, tpr, thresholds = roc_curve(all_targets, all_sims)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    opt_threshold = thresholds[eer_idx]
    binary_preds = (all_sims >= opt_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds).ravel()
    total = tp + tn + fp + fn
    far = fp / (fp + tn) * 100 if fp + tn > 0 else 0.0
    frr = fn / (fn + tp) * 100 if fn + tp > 0 else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall_metric = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * (precision * recall_metric) / (precision + recall_metric) if precision + recall_metric > 0 else 0.0
    report = f'========================================\n ASYMMETRIC RERANKING PIPELINE RESULTS (1-to-N CALIBRATED)\n========================================\n Total Queries: {total}\n 1-to-N AUROC : {roc_auc:.4f}\n 1-to-N Threshold : {opt_threshold:.4f}\n \n True Positives  : {tp}\n True Negatives  : {tn}\n False Positives : {fp}\n False Negatives : {fn}\n\n--- FINAL PERFORMANCE ---\n False Acceptance Rate (FAR/FPR): {far:.2f}%\n False Rejection Rate  (FRR/FNR): {frr:.2f}%\n F1-Score: {f1:.4f}\n\n--- LATENCY BENCHMARKS ---\n Average End-to-End Latency     : {np.mean(end_to_end_latencies):.2f} ms\n========================================\n'
    with open(HISTORY_ROOT / '01_final_pipeline_report.txt', 'w') as f:
        f.write(report)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Final 1-to-N Pipeline ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(HISTORY_ROOT / 'roc_curve_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(6, 5))
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='flare', xticklabels=['Original', 'Plagiarism'], yticklabels=['Original', 'Plagiarism'], annot_kws={'size': 14})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Final Confusion Matrix (FAR: {far:.2f}%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(HISTORY_ROOT / 'confusion_matrix_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(8, 5))
    sns.histplot(end_to_end_latencies, bins=50, color='
    plt.axvline(np.mean(end_to_end_latencies), color='red', linestyle='--', label=f'Mean: {np.mean(end_to_end_latencies):.2f}ms')
    plt.axvline(np.percentile(end_to_end_latencies, 95), color='orange', linestyle=':', label=f'p95: {np.percentile(end_to_end_latencies, 95):.2f}ms')
    plt.xlabel('Latency (ms)', fontsize=12)
    plt.ylabel('Query Count', fontsize=12)
    plt.title('End-to-End Latency Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(HISTORY_ROOT / 'latency_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == '__main__':
    run_asymmetric_pipeline()