"""
Thesis-Quality Visualization Script
Stage 1 & Stage 2 — Chapter 4 Figures
White background, publication-ready, 300 DPI
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ── Output directories ──────────────────────────────────────────────────────
STAGE1_OUT = r"d:\RESEARCH PNGS\Thesis Figures\Stage1"
STAGE2_OUT = r"d:\RESEARCH PNGS\Thesis Figures\Stage2"
os.makedirs(STAGE1_OUT, exist_ok=True)
os.makedirs(STAGE2_OUT, exist_ok=True)

# ── Global Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#CCCCCC',
})

# ── Color Palettes ────────────────────────────────────────────────────────────
MODEL_COLORS = {
    'ResNet50':       '#2E86AB',
    'EfficientNet-B0': '#A23B72',
    'MobileNetV2':    '#F18F01',
    'ConvNeXt-Tiny':  '#48CAE4',
    'Swin-Tiny':      '#06A77D',
    'Ensemble':       '#E63946',
    'ENSEMBLE':       '#E63946',
    'ENSEMBLE*':    '#E63946',
}
CLASS_COLORS = {
    'MEL': '#C1121F',
    'BCC': '#2196F3',
    'SCC': '#4CAF50',
    'AK':  '#FF9800',
    'Benign':    '#4CAF50',
    'Malignant': '#C1121F',
}
METRIC_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#48CAE4', '#06A77D']

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def fig_s1_1_class_distribution():
    """Stage 1 — Fig 1: Dataset Class Distribution (Train & Test)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Stage 1: Dataset Class Distribution', fontsize=15, fontweight='bold', y=1.01)

    # Original training set (8 classes from ISIC 2019)
    train_classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    train_counts  = [4522, 12875, 3323, 867, 2624, 239, 253, 628]
    colors_train = ['#C1121F', '#4CAF50', '#2196F3', '#FF9800',
                    '#9C27B0', '#795548', '#00BCD4', '#FF5722']

    ax = axes[0]
    bars = ax.bar(train_classes, train_counts, color=colors_train, edgecolor='white',
                  linewidth=0.8, zorder=3, width=0.65)
    for bar, cnt in zip(bars, train_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                f'{cnt:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title('Original Training Set (25,331 images)', pad=10)
    ax.set_ylabel('Number of Images')
    ax.set_ylim(0, max(train_counts)*1.15)
    ax.set_xlabel('Skin Lesion Class')

    # After undersampling NV — binary split
    ax2 = axes[1]
    splits = ['Train\n(17,731)', 'Validation\n(3,800)', 'Test\n(3,800)']
    benign_counts  = [10967, 2351, 2348]
    malignant_counts = [6764, 1449, 1452]
    x = np.arange(len(splits))
    w = 0.35
    b1 = ax2.bar(x - w/2, benign_counts,   w, label='Benign',    color='#4CAF50', edgecolor='white', zorder=3)
    b2 = ax2.bar(x + w/2, malignant_counts, w, label='Malignant', color='#C1121F', edgecolor='white', zorder=3)
    for bars in [b1, b2]:
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                     f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_title('Binary Label Distribution Across Splits', pad=10)
    ax2.set_ylabel('Number of Images')
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits)
    ax2.set_ylim(0, max(benign_counts)*1.18)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    out = os.path.join(STAGE1_OUT, 'fig1_class_distribution.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


def fig_s1_2_model_comparison():
    """Stage 1 — Fig 2: Individual Model Performance Comparison (5 metrics)"""
    models = ['ResNet50', 'EfficientNet-B0', 'MobileNetV2', 'ConvNeXt-Tiny', 'Swin-Tiny']
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'F1-Score', 'AUC']
    data = {
        'ResNet50':       [0.9276, 0.9234, 0.9207, 0.9220, 0.9675],
        'EfficientNet-B0':[0.9287, 0.9204, 0.9287, 0.9242, 0.9713],
        'MobileNetV2':    [0.9189, 0.9108, 0.9168, 0.9136, 0.9680],
        'ConvNeXt-Tiny':  [0.9318, 0.9226, 0.9352, 0.9279, 0.9746],
        'Swin-Tiny':      [0.9355, 0.9271, 0.9374, 0.9316, 0.9734],
    }

    x = np.arange(len(metrics))
    w = 0.15
    offsets = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * w

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(models):
        vals = data[model]
        color = MODEL_COLORS[model]
        bars = ax.bar(x + offsets[i], vals, w, label=model,
                      color=color, edgecolor='white', linewidth=0.6, zorder=3)

    ax.set_title('Stage 1: Individual Model Performance on Validation Set', pad=12)
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0.88, 1.00)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.legend(loc='lower right', fontsize=9, ncol=3)

    # Highlight best per metric
    for j, metric in enumerate(metrics):
        vals_per_metric = [data[m][j] for m in models]
        best_val = max(vals_per_metric)
        best_idx = vals_per_metric.index(best_val)
        x_pos = j + offsets[best_idx]
        ax.annotate('*', (x_pos, best_val + 0.001), ha='center', va='bottom',
                    fontsize=9, color='#FFD700')

    plt.tight_layout()
    out = os.path.join(STAGE1_OUT, 'fig2_model_comparison.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


def fig_s1_3_ensemble_gain():
    """Stage 1 — Fig 3: Ensemble vs Best Individual (proper labels)"""
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'F1-Score', 'AUC']
    best_individual = [0.9355, 0.9271, 0.9374, 0.9316, 0.9746]  # best per metric
    ensemble        = [0.9434, 0.9112, 0.9379, 0.9244, 0.9845]
    best_model_name = ['Swin-Tiny', 'Swin-Tiny', 'Swin-Tiny', 'Swin-Tiny', 'ConvNeXt-Tiny']

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, best_individual, w, label='Best Individual Model',
                color='#5C6BC0', edgecolor='white', linewidth=0.7, zorder=3)
    b2 = ax.bar(x + w/2, ensemble,        w, label='Ensemble (5 Models)',
                color='#E63946', edgecolor='white', linewidth=0.7, zorder=3)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0008,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8.5, color='#333')
    for bar, gain, m_name in zip(b2, [e - b for e, b in zip(ensemble, best_individual)], best_model_name):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0008,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8.5, color='#333')
        if gain > 0:
            ax.annotate(f'^ +{gain:.4f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004),
                        ha='center', fontsize=8, color='#E63946', fontweight='bold')

    ax.set_title('Stage 1: Ensemble vs. Best Individual Model Performance', pad=12)
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0.88, 1.01)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    out = os.path.join(STAGE1_OUT, 'fig3_ensemble_gain.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


def fig_s1_4_summary_table():
    """Stage 1 — Fig 4: Professional summary metrics table"""
    models  = ['ResNet50', 'EfficientNet-B0', 'MobileNetV2', 'ConvNeXt-Tiny', 'Swin-Tiny', 'ENSEMBLE']
    headers = ['Model', 'Accuracy', 'Precision', 'Sensitivity', 'F1-Score', 'AUC']
    rows = [
        ['ResNet50',       '92.76%', '92.34%', '92.07%', '92.20%', '0.9675'],
        ['EfficientNet-B0','92.87%', '92.04%', '92.87%', '92.42%', '0.9713'],
        ['MobileNetV2',    '91.89%', '91.08%', '91.68%', '91.36%', '0.9680'],
        ['ConvNeXt-Tiny',  '93.18%', '92.26%', '93.52%', '92.79%', '0.9746'],
        ['Swin-Tiny',      '93.55%', '92.71%', '93.74%', '93.16%', '0.9734'],
        ['ENSEMBLE*',     '94.34%', '91.12%', '93.79%', '92.44%', '0.9845'],
    ]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis('off')
    ax.set_title('Stage 1: Complete Performance Summary', fontsize=14, fontweight='bold', pad=18)

    col_widths = [0.22, 0.13, 0.13, 0.16, 0.13, 0.13]
    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor('#2E3A59')
        cell.set_text_props(color='white', fontweight='bold')

    # Style rows
    for i, row in enumerate(rows, start=1):
        is_ensemble = '*' in row[0]
        for j in range(len(headers)):
            cell = table[i, j]
            if is_ensemble:
                cell.set_facecolor('#FFF3CD')
                cell.set_text_props(fontweight='bold', color='#7D3C00')
            elif i % 2 == 0:
                cell.set_facecolor('#F0F4FF')
            else:
                cell.set_facecolor('#FAFAFA')

        # Highlight best accuracy per column (skip ensemble for fair comparison)
        if not is_ensemble:
            for j, metric in enumerate(['Accuracy','Precision','Sensitivity','F1-Score','AUC'], start=1):
                pass  # handled below

    # Highlight per-column best (excluding ensemble row)
    col_idx_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    metric_vals = {
        1: [0.9276, 0.9287, 0.9189, 0.9318, 0.9355],
        2: [0.9234, 0.9204, 0.9108, 0.9226, 0.9271],
        3: [0.9207, 0.9287, 0.9168, 0.9352, 0.9374],
        4: [0.9220, 0.9242, 0.9136, 0.9279, 0.9316],
        5: [0.9675, 0.9713, 0.9680, 0.9746, 0.9734],
    }
    for col_j, vals in metric_vals.items():
        best_row = vals.index(max(vals)) + 1
        cell = table[best_row, col_j]
        cell.set_facecolor('#D5F5E3')
        cell.set_text_props(fontweight='bold', color='#1E8449')

    plt.tight_layout()
    out = os.path.join(STAGE1_OUT, 'fig4_summary_table.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


def fig_s1_5_pipeline_diagram():
    """Stage 1 — Fig 5: Two-stage pipeline architecture diagram"""
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.axis('off')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.set_title('Two-Stage Skin Lesion Classification Pipeline', fontsize=15,
                 fontweight='bold', pad=12)

    def box(x, y, w, h, color, text, fontsize=9.5, text_color='white'):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                                        facecolor=color, edgecolor='#333333',
                                        linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight='bold',
                multialignment='center', zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333333',
                                   lw=1.8, connectionstyle='arc3,rad=0'))

    # Input
    box(0.2, 1.8, 1.6, 1.4, '#546E7A', 'ISIC 2019\nDermoscopy\nImage', fontsize=9)
    arrow(1.8, 2.5, 2.3, 2.5)

    # Preprocessing
    box(2.3, 1.8, 1.8, 1.4, '#37474F', 'Artifact\nRemoval &\nResize 224px', fontsize=9)
    arrow(4.1, 2.5, 4.6, 2.5)

    # Stage 1 box
    box(4.6, 0.6, 3.2, 3.8, '#1565C0', 'STAGE 1\nBinary Classifier\n─────────────\n5-Model Ensemble\n(ResNet50, EffNet-B0,\nMobileNetV2, ConvNeXt,\nSwin-Tiny)\n─────────────\nAcc: 94.34%', fontsize=8.5)

    # Stage 1 outputs
    arrow(7.8, 3.5, 8.3, 4.0)
    box(8.3, 3.5, 2.0, 1.1, '#388E3C', 'BENIGN\n→ No further\n   action', fontsize=8.5)

    arrow(7.8, 1.5, 8.3, 1.5)
    box(8.3, 0.8, 2.0, 1.4, '#B71C1C', 'MALIGNANT\n→ Routed to\n   Stage 2', fontsize=8.5)

    arrow(10.3, 1.5, 10.8, 1.5)

    # Stage 2 box
    box(10.8, 0.4, 3.0, 4.2, '#6A1B9A', 'STAGE 2\n4-Class Classifier\n─────────────\n5-Model Ensemble\n(MEL / BCC / SCC / AK)\n─────────────\nAcc: 85.58%\nSCC Recall: 95.79%', fontsize=8.5)

    arrow(13.8, 2.5, 14.2, 2.5)
    box(14.2, 1.8, 0.7, 1.4, '#4A148C', 'Final\nDiag.', fontsize=8, text_color='white')

    plt.tight_layout()
    out = os.path.join(STAGE1_OUT, 'fig5_pipeline_diagram.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def fig_s2_1_class_distribution():
    """Stage 2 — Fig 1: Raw & Balanced Malignant Class Distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Stage 2: Malignant Sub-Class Distribution', fontsize=15, fontweight='bold', y=1.02)

    classes = ['MEL', 'BCC', 'SCC', 'AK']
    raw_counts = [4522, 3323, 628, 867]
    balanced   = [2000, 2000, 628, 867]
    colors = [CLASS_COLORS[c] for c in classes]

    # Before
    ax = axes[0]
    bars = ax.bar(classes, raw_counts, color=colors, edgecolor='white', linewidth=0.8,
                  zorder=3, width=0.55)
    for bar, cnt in zip(bars, raw_counts):
        pct = cnt / sum(raw_counts) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{cnt:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax.set_title('Before Preprocessing\n(Raw Training Set)', pad=10)
    ax.set_ylabel('Number of Images')
    ax.set_ylim(0, 5500)
    ax.set_xlabel('Malignant Subclass')

    # After
    ax2 = axes[1]
    bars2 = ax2.bar(classes, balanced, color=colors, edgecolor='white', linewidth=0.8,
                    zorder=3, width=0.55, alpha=0.9)
    ax2.axhline(y=2000, color='#E63946', linestyle='--', linewidth=1.5,
                label='Undersampling cap = 2,000', zorder=4)
    for bar, cnt in zip(bars2, balanced):
        pct = cnt / sum(balanced) * 100
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                 f'{cnt:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax2.set_title('After Majority-Class Undersampling\n(Cap = 2,000 per class)', pad=10)
    ax2.set_ylabel('Number of Images')
    ax2.set_ylim(0, 2800)
    ax2.set_xlabel('Malignant Subclass')
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    out = os.path.join(STAGE2_OUT, 'fig1_class_distribution.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


def fig_s2_2_model_comparison():
    """Stage 2 — Fig 2: Individual Model Performance (all key metrics)"""
    models = ['ResNet50', 'EfficientNet-B0', 'MobileNetV2', 'ConvNeXt-Tiny', 'Swin-Tiny']
    metrics = ['Accuracy', 'Macro\nPrecision', 'Macro\nSensitivity', 'Macro\nF1', 'Macro\nAUC']
    data = {
        'ResNet50':       [0.8048, 0.7665, 0.8188, 0.7814, 0.9481],
        'EfficientNet-B0':[0.8339, 0.7999, 0.8334, 0.8133, 0.9615],
        'MobileNetV2':    [0.8024, 0.7618, 0.8044, 0.7764, 0.9506],
        'ConvNeXt-Tiny':  [0.8036, 0.7647, 0.8190, 0.7789, 0.9549],
        'Swin-Tiny':      [0.8291, 0.7903, 0.8441, 0.8063, 0.9617],
    }

    x = np.arange(len(metrics))
    w = 0.15
    offsets = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * w

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(models):
        color = MODEL_COLORS[model]
        bars = ax.bar(x + offsets[i], data[model], w, label=model,
                      color=color, edgecolor='white', linewidth=0.6, zorder=3)

    ax.set_title('Stage 2: Individual Model Performance on Test Set', pad=12)
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0.70, 1.00)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.legend(loc='lower right', fontsize=9, ncol=3)

    # Mark best per metric with *
    for j in range(len(metrics)):
        vals = [data[m][j] for m in models]
        best_i = vals.index(max(vals))
        ax.annotate('*', (j + offsets[best_i], vals[best_i] + 0.003),
                    ha='center', fontsize=9, color='#FFD700')

    plt.tight_layout()
    out = os.path.join(STAGE2_OUT, 'fig2_model_comparison.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


def fig_s2_3_per_class_recall():
    """Stage 2 — Fig 3: Per-class Recall for all models (clinical focus)"""
    models = ['ResNet50', 'EfficientNet-B0', 'MobileNetV2', 'ConvNeXt-Tiny', 'Swin-Tiny', 'ENSEMBLE']
    class_recalls = {
        'MEL':  [0.8167, 0.8600, 0.8233, 0.8067, 0.8300, 0.8500],
        'BCC':  [0.7733, 0.8200, 0.7900, 0.7767, 0.8000, 0.8567],
        'SCC':  [0.9158, 0.8842, 0.8737, 0.9158, 0.9158, 0.9579],
        'AK':   [0.7692, 0.7692, 0.7308, 0.7769, 0.8308, 0.7923],
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    fig.suptitle('Stage 2: Per-Class Recall — All Models vs. Ensemble', fontsize=14,
                 fontweight='bold', y=1.02)

    for ax, cls in zip(axes, ['MEL', 'BCC', 'SCC', 'AK']):
        recalls = class_recalls[cls]
        bar_colors = [MODEL_COLORS.get(m, '#999') for m in models]
        # Make ensemble stand out
        bar_colors[-1] = '#E63946'
        alphas = [0.8]*5 + [1.0]

        bars = ax.bar(range(len(models)), recalls, color=bar_colors,
                      edgecolor='white', linewidth=0.7, zorder=3, width=0.65)
        for bar, val, alpha in zip(bars, recalls, alphas):
            bar.set_alpha(alpha)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        ax.set_title(f'{cls}\nRecall', pad=8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(['RN50', 'EffB0', 'MNV2', 'CNX', 'Swin', 'ENS'],
                           fontsize=8.5, rotation=30, ha='right')
        ax.set_ylim(0.65, 1.02)
        ax.set_ylabel('Recall' if cls == 'MEL' else '')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f}'))

        # Highlight clinical concern (SCC is most critical)
        if cls == 'SCC':
            ax.set_title(f'{cls} Recall\n(* Highest Clinical Priority)', pad=8)
            ax.spines['bottom'].set_color('#E63946')
            ax.spines['bottom'].set_linewidth(2.0)
            ax.spines['left'].set_color('#E63946')
            ax.spines['left'].set_linewidth(2.0)

    plt.tight_layout()
    out = os.path.join(STAGE2_OUT, 'fig3_per_class_recall.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


def fig_s2_4_ensemble_gain():
    """Stage 2 — Fig 4: Ensemble vs Best Individual (all metrics)"""
    metrics = ['Accuracy', 'Macro Precision', 'Macro Sensitivity', 'Macro F1', 'Macro AUC']
    best_individual = [0.8339, 0.7999, 0.8441, 0.8133, 0.9617]  # per-metric best
    ensemble        = [0.8558, 0.8168, 0.8642, 0.8332, 0.9738]
    best_labels     = ['EfficientNet-B0', 'EfficientNet-B0', 'Swin-Tiny',
                       'EfficientNet-B0', 'Swin-Tiny']

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - w/2, best_individual, w, label='Best Individual Model',
                color='#5C6BC0', edgecolor='white', linewidth=0.7, zorder=3)
    b2 = ax.bar(x + w/2, ensemble,        w, label='Ensemble (5 Models)',
                color='#E63946', edgecolor='white', linewidth=0.7, zorder=3)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8, color='#333')
    for bar, gain in zip(b2, [e - b for e, b in zip(ensemble, best_individual)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8, color='#333')
        if gain > 0:
            ax.annotate(f'^ +{gain:.4f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01),
                        ha='center', fontsize=8.5, color='#B71C1C', fontweight='bold')

    ax.set_title('Stage 2: Ensemble vs. Best Individual Model Performance', pad=12)
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10.5)
    ax.set_ylim(0.75, 1.01)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    out = os.path.join(STAGE2_OUT, 'fig4_ensemble_gain.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


def fig_s2_5_radar():
    """Stage 2 — Fig 5: Radar chart — all 6 models clearly distinguishable"""
    from matplotlib.patches import FancyArrowPatch

    categories = ['Accuracy', 'Macro\nPrecision', 'Macro\nSensitivity',
                  'Macro F1', 'MEL\nRecall', 'BCC\nRecall', 'SCC\nRecall', 'AK\nRecall']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    models_data = {
        'ResNet50':       [0.8048, 0.7665, 0.8188, 0.7814, 0.8167, 0.7733, 0.9158, 0.7692],
        'EfficientNet-B0':[0.8339, 0.7999, 0.8334, 0.8133, 0.8600, 0.8200, 0.8842, 0.7692],
        'MobileNetV2':    [0.8024, 0.7618, 0.8044, 0.7764, 0.8233, 0.7900, 0.8737, 0.7308],
        'ConvNeXt-Tiny':  [0.8036, 0.7647, 0.8190, 0.7789, 0.8067, 0.7767, 0.9158, 0.7769],
        'Swin-Tiny':      [0.8291, 0.7903, 0.8441, 0.8063, 0.8300, 0.8000, 0.9158, 0.8308],
        'ENSEMBLE':       [0.8558, 0.8168, 0.8642, 0.8332, 0.8500, 0.8567, 0.9579, 0.7923],
    }
    linestyles = ['-', '--', ':', '-.', (0, (3,1,1,1)), '-']
    linewidths = [1.5, 1.5, 1.5, 1.5, 1.5, 2.8]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')

    for i, (model, vals) in enumerate(models_data.items()):
        color = MODEL_COLORS.get(model, '#999')
        lw    = linewidths[i]
        ls    = linestyles[i]
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, color=color, linewidth=lw, linestyle=ls, label=model, zorder=3)
        if model == 'ENSEMBLE':
            ax.fill(angles, vals_closed, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10.5, fontweight='bold')
    ax.set_ylim(0.60, 1.0)
    ax.set_yticks([0.65, 0.75, 0.85, 0.95])
    ax.set_yticklabels(['0.65', '0.75', '0.85', '0.95'], size=9, color='#666')
    ax.grid(color='#CCCCCC', linewidth=0.6, linestyle='--')
    ax.set_title('Stage 2: Multi-Model Performance Radar\n(All Metrics)', size=13,
                 fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.15), fontsize=10,
              framealpha=0.95, edgecolor='#CCC')

    plt.tight_layout()
    out = os.path.join(STAGE2_OUT, 'fig5_radar_chart.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


def fig_s2_6_confusion_matrix_ensemble():
    """Stage 2 — Fig 6: Ensemble confusion matrix (large, clear, standalone)"""
    # Normalized confusion matrix values from existing confusion_matrices.png
    cm_norm = np.array([
        [0.85, 0.04, 0.05, 0.06],  # MEL
        [0.01, 0.86, 0.05, 0.09],  # BCC (was 0.86)  — from existing plot
        [0.00, 0.02, 0.96, 0.02],  # SCC
        [0.04, 0.08, 0.08, 0.79],  # AK
    ])
    labels = ['MEL', 'BCC', 'SCC', 'AK']

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.Blues
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Normalized Count')

    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Stage 2 Ensemble — Normalized Confusion Matrix\n(Test Set)', fontsize=13,
                 fontweight='bold', pad=12)
    ax.grid(False)
    ax.spines[:].set_visible(False)

    thresh = 0.5
    for i in range(4):
        for j in range(4):
            val = cm_norm[i, j]
            color = 'white' if val > thresh else 'black'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=13, fontweight=weight)

    plt.tight_layout()
    out = os.path.join(STAGE2_OUT, 'fig6_ensemble_confusion_matrix.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


def fig_s2_7_all_confusion_matrices():
    """Stage 2 — Fig 7: All 5 models + Ensemble confusion matrices (2-row grid)"""
    cms = {
        'ResNet50':       np.array([[0.82,0.05,0.05,0.08],[0.03,0.77,0.07,0.13],[0.01,0.02,0.92,0.05],[0.05,0.08,0.11,0.77]]),
        'EfficientNet-B0':np.array([[0.86,0.04,0.03,0.06],[0.03,0.82,0.04,0.11],[0.03,0.06,0.88,0.02],[0.06,0.12,0.05,0.77]]),
        'MobileNetV2':    np.array([[0.82,0.06,0.05,0.06],[0.03,0.79,0.06,0.11],[0.04,0.03,0.87,0.05],[0.05,0.13,0.09,0.73]]),
        'ConvNeXt-Tiny':  np.array([[0.81,0.05,0.08,0.06],[0.02,0.78,0.10,0.11],[0.03,0.02,0.92,0.03],[0.05,0.09,0.08,0.78]]),
        'Swin-Tiny':      np.array([[0.83,0.06,0.05,0.07],[0.01,0.80,0.08,0.11],[0.00,0.04,0.92,0.04],[0.03,0.05,0.09,0.83]]),
        'ENSEMBLE':       np.array([[0.85,0.04,0.05,0.06],[0.01,0.86,0.05,0.09],[0.00,0.02,0.96,0.02],[0.04,0.08,0.08,0.79]]),
    }
    labels = ['MEL', 'BCC', 'SCC', 'AK']
    model_names = list(cms.keys())
    accs = [0.8048, 0.8339, 0.8024, 0.8036, 0.8291, 0.8558]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Stage 2: Normalized Confusion Matrices — All Models', fontsize=15,
                 fontweight='bold', y=1.01)
    axes_flat = axes.flatten()

    cmap = plt.cm.Blues
    for ax, (model, cm), acc in zip(axes_flat, cms.items(), accs):
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)

        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
        ax.set_ylabel('True', fontsize=10, fontweight='bold')
        ax.grid(False)
        ax.spines[:].set_visible(False)

        title_color = '#C62828' if model == 'ENSEMBLE' else 'black'
        weight = 'bold' if model == 'ENSEMBLE' else 'normal'
        ax.set_title(f'{"* " if model=="ENSEMBLE" else ""}{model}\n(Acc = {acc:.4f})',
                     fontsize=11, fontweight=weight, color=title_color, pad=8)

        thresh = 0.5
        for i in range(4):
            for j in range(4):
                val = cm[i, j]
                color = 'white' if val > thresh else 'black'
                fw = 'bold' if i == j else 'normal'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=11, fontweight=fw)

    plt.colorbar(im, ax=axes_flat[-1], fraction=0.046, pad=0.04, label='Normalized Count')
    plt.tight_layout()
    out = os.path.join(STAGE2_OUT, 'fig7_all_confusion_matrices.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


def fig_s2_8_summary_table():
    """Stage 2 — Fig 8: Professional summary table"""
    headers = ['Model', 'Accuracy', 'Macro\nPrecision', 'Macro\nSensitivity',
               'Macro F1', 'Macro AUC', 'MEL\nRecall', 'BCC\nRecall', 'SCC\nRecall', 'AK\nRecall']
    rows = [
        ['ResNet50',        '80.48%','76.65%','81.88%','78.14%','0.9481','81.67%','77.33%','91.58%','76.92%'],
        ['EfficientNet-B0', '83.39%','79.99%','83.34%','81.33%','0.9615','86.00%','82.00%','88.42%','76.92%'],
        ['MobileNetV2',     '80.24%','76.18%','80.44%','77.64%','0.9506','82.33%','79.00%','87.37%','73.08%'],
        ['ConvNeXt-Tiny',   '80.36%','76.47%','81.90%','77.89%','0.9549','80.67%','77.67%','91.58%','77.69%'],
        ['Swin-Tiny',       '82.91%','79.03%','84.41%','80.63%','0.9617','83.00%','80.00%','91.58%','83.08%'],
        ['ENSEMBLE*',      '85.58%','81.68%','86.42%','83.32%','0.9738','85.00%','85.67%','95.79%','79.23%'],
    ]

    fig, ax = plt.subplots(figsize=(18, 4.5))
    ax.axis('off')
    ax.set_title('Stage 2: Complete Performance Summary (Test Set)', fontsize=14,
                 fontweight='bold', pad=18)

    col_widths = [0.14] + [0.095]*9
    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc='center', loc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.2)

    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor('#2E3A59')
        cell.set_text_props(color='white', fontweight='bold')

    for i, row in enumerate(rows, start=1):
        is_ensemble = '*' in row[0]
        for j in range(len(headers)):
            cell = table[i, j]
            if is_ensemble:
                cell.set_facecolor('#FFF3CD')
                cell.set_text_props(fontweight='bold', color='#7D3C00')
            elif i % 2 == 0:
                cell.set_facecolor('#F0F4FF')
            else:
                cell.set_facecolor('#FAFAFA')

    # Highlight best per column (excluding ensemble)
    col_vals = {
        1: [0.8048, 0.8339, 0.8024, 0.8036, 0.8291],
        2: [0.7665, 0.7999, 0.7618, 0.7647, 0.7903],
        3: [0.8188, 0.8334, 0.8044, 0.8190, 0.8441],
        4: [0.7814, 0.8133, 0.7764, 0.7789, 0.8063],
        5: [0.9481, 0.9615, 0.9506, 0.9549, 0.9617],
        6: [0.8167, 0.8600, 0.8233, 0.8067, 0.8300],
        7: [0.7733, 0.8200, 0.7900, 0.7767, 0.8000],
        8: [0.9158, 0.8842, 0.8737, 0.9158, 0.9158],
        9: [0.7692, 0.7692, 0.7308, 0.7769, 0.8308],
    }
    for col_j, vals in col_vals.items():
        best_row = vals.index(max(vals)) + 1
        cell = table[best_row, col_j]
        cell.set_facecolor('#D5F5E3')
        cell.set_text_props(fontweight='bold', color='#1E8449')

    plt.tight_layout()
    out = os.path.join(STAGE2_OUT, 'fig8_summary_table.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


def fig_s2_9_clinical_bar():
    """Stage 2 — Fig 9: Clinical Priority Score bar chart (clean, colored)"""
    models = ['ResNet50', 'EfficientNet-B0', 'MobileNetV2', 'ConvNeXt-Tiny', 'Swin-Tiny', 'ENSEMBLE']
    scores = [0.8472, 0.8566, 0.8309, 0.8443, 0.8614, 0.8881]
    colors = [MODEL_COLORS.get(m, '#999') for m in models]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(models[::-1], scores[::-1], color=colors[::-1],
                   edgecolor='white', linewidth=0.8, height=0.55, zorder=3)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center', ha='left', fontsize=10.5, fontweight='bold')

    ax.set_xlim(0.78, 0.92)
    ax.set_xlabel('Clinical Priority Score\n(Weighted: 40% MEL + 40% SCC + 20% BCC Recall + 0% AK)', fontsize=10)
    ax.set_title('Stage 2: Clinical Priority Score by Model', fontsize=13, fontweight='bold', pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f}'))

    # Highlight ENSEMBLE
    bars[0].set_edgecolor('#B71C1C')
    bars[0].set_linewidth(2.0)

    plt.tight_layout()
    out = os.path.join(STAGE2_OUT, 'fig9_clinical_priority.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


def fig_s2_10_gradcam_composite():
    """Stage 2 — Fig 10: Grad-CAM composite (load from existing files, add labels)"""
    scc_path = r"d:\RESEARCH PNGS\Final Stage 2\grad_cam_SCC.png"
    ak_path  = r"d:\RESEARCH PNGS\Final Stage 2\grad_cam_AK.png"

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle('Stage 2 Ensemble: Grad-CAM Explainability Visualizations',
                 fontsize=15, fontweight='bold', y=1.01)

    for ax, path, cls_name, cls_color in zip(
            axes,
            [scc_path, ak_path],
            ['SCC (Squamous Cell Carcinoma)', 'AK (Actinic Keratosis)'],
            ['#4CAF50', '#FF9800']):
        img = np.array(Image.open(path))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'{cls_name}  |  Original (top row) + Grad-CAM Overlay (bottom row)',
                     fontsize=11.5, fontweight='bold', color=cls_color,
                     loc='left', pad=8)

    plt.tight_layout()
    out = os.path.join(STAGE2_OUT, 'fig10_gradcam_composite.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


def fig_s1_gradcam_fixed():
    """Stage 1 — Grad-CAM composite with fixed labels"""
    src_path = r"d:\RESEARCH PNGS\Final stage1\gradcam_comparison_summary.png"
    img = np.array(Image.open(src_path))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Stage 1: Grad-CAM Heatmaps per Model — Malignant Lesion Sample\n'
                 '(ResNet50 · EfficientNet-B0 · MobileNetV2 · ConvNeXt-Tiny · Swin-Tiny)',
                 fontsize=13, fontweight='bold', pad=10)

    # Add model labels manually below each panel
    fig.text(0.10, 0.01, 'ResNet50',        ha='center', fontsize=10, fontweight='bold', color='#2E86AB')
    fig.text(0.30, 0.01, 'EfficientNet-B0', ha='center', fontsize=10, fontweight='bold', color='#A23B72')
    fig.text(0.50, 0.01, 'MobileNetV2',     ha='center', fontsize=10, fontweight='bold', color='#F18F01')
    fig.text(0.70, 0.01, 'ConvNeXt-Tiny',  ha='center', fontsize=10, fontweight='bold', color='#48CAE4')
    fig.text(0.90, 0.01, 'Swin-Tiny',      ha='center', fontsize=10, fontweight='bold', color='#06A77D')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(STAGE1_OUT, 'fig6_gradcam.png')
    plt.savefig(out); plt.close()
    print(f"  [OK] Saved: {os.path.basename(out)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  GENERATING THESIS FIGURES - STAGE 1")
    print("="*60)
    fig_s1_1_class_distribution()
    fig_s1_2_model_comparison()
    fig_s1_3_ensemble_gain()
    fig_s1_4_summary_table()
    fig_s1_5_pipeline_diagram()
    fig_s1_gradcam_fixed()

    print("\n" + "="*60)
    print("  GENERATING THESIS FIGURES - STAGE 2")
    print("="*60)
    fig_s2_1_class_distribution()
    fig_s2_2_model_comparison()
    fig_s2_3_per_class_recall()
    fig_s2_4_ensemble_gain()
    fig_s2_5_radar()
    fig_s2_6_confusion_matrix_ensemble()
    fig_s2_7_all_confusion_matrices()
    fig_s2_8_summary_table()
    fig_s2_9_clinical_bar()
    fig_s2_10_gradcam_composite()

    print("\n" + "="*60)
    print("  ALL DONE!")
    print(f"  Stage 1 figures → {STAGE1_OUT}")
    print(f"  Stage 2 figures → {STAGE2_OUT}")
    print("="*60 + "\n")
