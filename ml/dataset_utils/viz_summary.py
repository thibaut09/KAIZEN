import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.simplefilter('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial']
plt.rcParams['axes.edgecolor'] = '
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.facecolor'] = '
plt.rcParams['axes.facecolor'] = '
plt.rcParams['savefig.facecolor'] = '
PROJECT_ROOT = Path('c:\\Users\\lolip\\Downloads\\NFT Originality Detection')
HISTORY_ROOT = PROJECT_ROOT / 'experiments' / 'history_strict_jury'
DEFENSE_DIR = HISTORY_ROOT / 'Thesis_Defense_Visuals'
DEFENSE_DIR.mkdir(parents=True, exist_ok=True)
labels = ['Exp 1\nDINOv2', 'Exp 2\nSigLIP2', 'Exp 3\nLate Fusion', 'Exp 4\nProj. Head', 'Exp 6\nOracle']
aurocs = [0.118, 0.661, 0.382, 0.986, 0.997]
latencies = [135, 120, 260, 45, 5.8]

def generate_simple_auroc_bar():
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette('RdYlBu', 5)
    bars = ax.bar(labels, aurocs, color=colors, width=0.55, edgecolor='
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('AUROC Score', fontsize=15, fontweight='bold', labelpad=15)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold', color='
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(DEFENSE_DIR / '01_Simple_AUROC_Bar.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_simple_latency_scatter():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(latencies, aurocs, marker='D', linestyle='-', color='
    ax.set_xscale('log')
    ax.set_xlabel('Average Query Latency (ms) [Log Scale]', fontsize=15, fontweight='bold', labelpad=15)
    ax.set_ylabel('AUROC Score', fontsize=15, fontweight='bold', labelpad=15)
    for i, txt in enumerate(labels):
        ax.text(latencies[i], aurocs[i] - 0.08, txt.replace('\n', ' '), ha='center', fontsize=12, fontweight='bold', color='
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(DEFENSE_DIR / '02_Simple_Latency_vs_AUROC.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_simple_line_progression():
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(labels))
    ax.plot(x_pos, aurocs, marker='o', linestyle='--', color='
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('AUROC Score', fontsize=15, fontweight='bold', labelpad=15)
    for i, txt in enumerate(aurocs):
        ax.text(x_pos[i], aurocs[i] + 0.035, f'{txt:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold', color='
    ax.tick_params(axis='y', labelsize=13)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(DEFENSE_DIR / '03_Simple_Line_Progression.png', dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == '__main__':
    generate_simple_auroc_bar()
    generate_simple_latency_scatter()
    generate_simple_line_progression()