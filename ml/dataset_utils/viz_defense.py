import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import matplotlib.patches as mpatches
warnings.simplefilter('ignore')
LIGHT_BLUE_BG = '
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial']
plt.rcParams['axes.edgecolor'] = '
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.facecolor'] = LIGHT_BLUE_BG
plt.rcParams['axes.facecolor'] = LIGHT_BLUE_BG
plt.rcParams['savefig.facecolor'] = LIGHT_BLUE_BG
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_ROOT = Path('c:\\Users\\lolip\\Downloads\\NFT Originality Detection')
HISTORY_ROOT = PROJECT_ROOT / 'experiments' / 'history_strict_jury'
DEFENSE_DIR = HISTORY_ROOT / 'Thesis_Defense_Visuals'
DEFENSE_DIR.mkdir(parents=True, exist_ok=True)

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

def generate_evolution_chart():
    experiments = ['Exp 1\nDINOv2', 'Exp 2\nSigLIP2', 'Exp 3\nLate Fusion', 'Exp 4\nProjection Head', 'Exp 5\nKaizen Oracle']
    aurocs = [0.11, 0.66, 0.38, 0.98, 0.997]
    colors = ['
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(experiments, aurocs, color=colors, width=0.6, edgecolor='black', linewidth=1)
    ax.plot(experiments, aurocs, color='
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel('AUROC (Area Under ROC)', fontsize=14, fontweight='bold')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.annotate('Catastrophic Failure\n(Sibling Paradox)', xy=(1, 0.11), xytext=(1.5, 0.25), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), fontsize=11, color='
    ax.annotate('Architectural\nBreakthrough', xy=(3, 0.98), xytext=(2.5, 0.85), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), fontsize=11, color='
    plt.tight_layout()
    plt.savefig(DEFENSE_DIR / '01_Evolution_Chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_tsne_comparison():
    FEATURES_DIR = PROJECT_ROOT / 'dataset' / 'extracted_features'
    WEIGHTS_FILE = PROJECT_ROOT / 'experiments' / 'history' / 'upgrade_v8_ablation' / 'v8_head_margin_0.70.pth'
    proj_head = projection_head().to(DEVICE)
    if os.path.exists(WEIGHTS_FILE):
        proj_head.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE, weights_only=True))
    proj_head.eval()
    batch_files = list(FEATURES_DIR.glob(f'dual_features_test_batch_*.pt'))
    if not batch_files:
        return
    pristine_dict = {}
    attack_dict = {}
    for bf in sorted(batch_files)[:2]:
        data = torch.load(bf, map_location='cpu', weights_only=False)['vanilla']
        for k, v in data.items():
            if '_attack_' in k:
                attack_dict[k] = v
            else:
                pristine_dict[k] = v
    from collections import defaultdict
    colls = defaultdict(list)
    for k in pristine_dict.keys():
        c = k.split('/')[0]
        colls[c].append(k)
    all_dino_vecs = []
    all_proj_vecs = []
    labels = []
    np.random.seed(42)
    count = 0
    for c, items in colls.items():
        if len(items) < 2:
            continue
        for orig_k in items:
            base = Path(orig_k).stem
            attacks = [ak for ak in attack_dict.keys() if ak.startswith(f'{c}/{base}_attack_')]
            if attacks:
                sib_candidates = [ik for ik in items if ik != orig_k]
                if not sib_candidates:
                    continue
                sib_k = np.random.choice(sib_candidates)
                atk_k = np.random.choice(attacks)
                all_dino_vecs.append(pristine_dict[orig_k][:768])
                t_orig = torch.tensor(pristine_dict[orig_k]).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    all_proj_vecs.append(proj_head(t_orig).cpu().numpy()[0])
                labels.append('Original (Authentic)')
                all_dino_vecs.append(pristine_dict[sib_k][:768])
                t_sib = torch.tensor(pristine_dict[sib_k]).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    all_proj_vecs.append(proj_head(t_sib).cpu().numpy()[0])
                labels.append('Sibling (Authentic)')
                all_dino_vecs.append(attack_dict[atk_k][:768])
                t_atk = torch.tensor(attack_dict[atk_k]).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    all_proj_vecs.append(proj_head(t_atk).cpu().numpy()[0])
                labels.append('Attack (Plagiarized)')
                count += 1
                if count >= 300:
                    break
        if count >= 300:
            break
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, init='pca', learning_rate='auto')
    dino_all = tsne.fit_transform(np.array(all_dino_vecs))
    proj_all = tsne.fit_transform(np.array(all_proj_vecs))
    palette = {'Original (Authentic)': '
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.scatterplot(x=dino_all[:, 0], y=dino_all[:, 1], hue=labels, palette=palette, alpha=0.8, ax=ax1, edgecolor='white', s=70)
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax1.legend(title='Asset Class', loc='best')
    sns.scatterplot(x=proj_all[:, 0], y=proj_all[:, 1], hue=labels, palette=palette, alpha=0.8, ax=ax2, edgecolor='white', s=70)
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax2.legend(title='Asset Class', loc='best')
    plt.tight_layout()
    plt.savefig(DEFENSE_DIR / '02_Latent_Space_Untangling.png', dpi=300, bbox_inches='tight')
    plt.close()

def _generate_synthetic_tsne():
    pass

def generate_performance_bubble():
    models = ['On-Chain ViT\n(Standard)', 'Experiment 5\n(Cascade)', 'Experiment 6\n(Asymmetric)']
    latencies = [30000, 70, 5.8]
    f1_scores = [0.66, 0.997, 0.9966]
    costs = [600, 0.9, 0.9]
    colors = ['
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axvline(x=15000, color='red', linestyle='--', linewidth=2, label='15,000ms NFR Limit')
    ax.axvspan(15000, 50000, color='red', alpha=0.05)
    ax.text(17000, 0.7, 'TIMEOUT ZONE\n(Oracle Fails)', color='red', fontweight='bold', fontsize=12)
    bubble_sizes = [c * 5 for c in costs]
    scatter = ax.scatter(latencies, f1_scores, s=bubble_sizes, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlim(1, 50000)
    ax.set_ylim(0.5, 1.05)
    ax.set_xlabel('End-to-End Latency (ms) [Log Scale]', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score (Accuracy)', fontsize=14, fontweight='bold')
    for i, txt in enumerate(models):
        y_offset = 0.03 if i != 1 else -0.04
        ax.text(latencies[i], f1_scores[i] + y_offset, txt, ha='center', fontweight='bold', fontsize=11)
    handles, labels = scatter.legend_elements(prop='sizes', alpha=0.6, num=3, func=lambda s: s / 5)
    labels = ['$0.90', '$300', '$600']
    legend2 = ax.legend(handles, labels, loc='lower left', title='Compute Cost / Query')
    ax.add_artist(legend2)
    import matplotlib.lines as mlines
    nfr_line = mlines.Line2D([], [], color='red', linestyle='--', label='15,000ms NFR Limit')
    ax.legend(handles=[nfr_line], loc='upper left')
    plt.tight_layout()
    plt.savefig(DEFENSE_DIR / '03_Performance_Latency_SweetSpot.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_waterfall_chart():
    stages = ['Standard ViT\n(10k NFTs)', 'Stage 0\nMetadata Gate', 'Stage 1/2\nOff-Chain AI', 'Kaizen Oracle\nFinal Cost']
    values = [6000000, -2874000, -3117000, 9000]
    totals = []
    current = 0
    for v in values:
        if v > 0 and current == 0:
            totals.append(0)
            current = v
        elif v < 0:
            totals.append(current)
            current += v
        else:
            totals.append(0)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['
    for i in range(len(stages)):
        val = values[i]
        start = totals[i]
        color = colors[i]
        if i == 0 or i == 3:
            ax.bar(stages[i], val, color=color, edgecolor='black', width=0.6)
            text_y = val + 100000
        else:
            ax.bar(stages[i], abs(val), bottom=start - abs(val), color=color, edgecolor='black', width=0.6)
            text_y = start - abs(val) - 250000
        text_val = f'${abs(val):,.0f}' if i == 0 or i == 3 else f'-${abs(val):,.0f}'
        ax.text(i, text_y, text_val, ha='center', fontweight='bold', fontsize=12, color=color if i != 0 and i != 3 else 'black')
    ax.set_ylabel('Operational Cost ($ USD)', fontsize=14, fontweight='bold')
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig(DEFENSE_DIR / '04_Economic_Impact_Waterfall.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_threshold_heatmap():
    thresholds = ['0.60', '0.65', '0.68\n(Optimal)', '0.70', '0.75']
    environments = ['Balanced\n(50% Attacks)', 'Clean-Heavy\n(10% Attacks)', 'Enterprise\n(1% Attacks)']
    matrix = np.array([[15.4, 4.2, 0.47, 0.12, 0.0], [22.8, 6.1, 0.55, 0.15, 0.01], [31.2, 8.4, 0.62, 0.18, 0.02]])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Reds', ax=ax, xticklabels=thresholds, yticklabels=environments, cbar_kws={'label': 'False Acceptance Rate (FAR) %'}, annot_kws={'fontsize': 14, 'fontweight': 'bold'}, linewidths=1, linecolor='black')
    for i in range(3):
        ax.add_patch(mpatches.Rectangle((2, i), 1, 1, fill=False, edgecolor='
    ax.set_xlabel('Classification Threshold', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Deployment Environment', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(DEFENSE_DIR / '05_Threshold_Sensitivity_Heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == '__main__':
    generate_evolution_chart()
    generate_tsne_comparison()
    generate_performance_bubble()
    generate_waterfall_chart()
    generate_threshold_heatmap()