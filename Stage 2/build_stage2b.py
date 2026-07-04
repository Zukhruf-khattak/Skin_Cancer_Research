"""
build_stage2b.py
Generates the improved Stage 2b Jupyter notebook:
  - Hard undersample MEL/BCC to MAX_MAJORITY=2000 images
  - WeightedRandomSampler on reduced dataset (no naive oversampling)
  - ImageNet pretrained weights (not from scratch)
  - Class-weighted Focal Loss (double safety net)
  - Richer augmentation
  - 50 epochs + early stopping
"""

import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

OUT_PATH = os.path.join(os.path.dirname(__file__), 'stage-2b-improved.ipynb')


def code(src_lines, cell_id):
    return {
        'cell_type': 'code',
        'id': cell_id,
        'metadata': {},
        'source': src_lines if isinstance(src_lines, list) else [src_lines],
        'outputs': [],
        'execution_count': None
    }


def md(src_lines, cell_id):
    return {
        'cell_type': 'markdown',
        'id': cell_id,
        'metadata': {},
        'source': src_lines if isinstance(src_lines, list) else [src_lines]
    }


# ─────────────────────────────────────────────────────────────────────────────
cells = []

# ── 0: Title ─────────────────────────────────────────────────────────────────
cells.append(md("""# Stage 2b: Malignant Subtype Classification — Improved
**ISIC 2019 Dataset | Pretrained Weights + Undersample MEL/BCC + WeightedRandomSampler + Class-Weighted Focal Loss**

This notebook improves upon Stage 2a by addressing the key weaknesses of that approach:

| Change | Stage 2a (old) | Stage 2b (improved) |
|--------|---------------|---------------------|
| Weights | From scratch | **ImageNet pretrained** |
| Majority classes | Full dataset (MEL 4522, BCC 3323) | **Hard-cap MEL/BCC to 2000** |
| Minority classes | None (shuffle only) | **WeightedRandomSampler** (no duplication) |
| Loss | Focal Loss only | **Focal Loss + Class Weights** |
| Augmentation | Basic resize | **Rich augmentation pipeline** |
| Epochs | 100 + early stop | **50 + early stop** |

## Strategy

1. **Hard undersample** MEL and BCC to 2000 images each (random subset, seed=42)
2. **WeightedRandomSampler** on the resulting dataset — SCC/AK appear more often per epoch, no files duplicated
3. **Class-weighted Focal Loss** — extra gradient signal for rare classes within each batch
""", 'cell_00'))

# ── 1: Imports ───────────────────────────────────────────────────────────────
cells.append(code("""\
import os, time, copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, accuracy_score,
                              precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
print('PyTorch version:', torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
CLASS_NAMES = ['MEL', 'BCC', 'SCC', 'AK']
NUM_CLASSES = 4
""", 'cell_01'))

# ── 2: Section 1 header ───────────────────────────────────────────────────────
cells.append(md("""\
## 1. Dataset Exploration & Class Imbalance

The ISIC 2019 malignant subset has severe class imbalance:
- **MEL**: ~48%  |  **BCC**: ~35%  |  **SCC**: ~6.7%  |  **AK**: ~9.3%

Stage 2b corrects this without ever duplicating image files.
""", 'cell_02'))

# ── 3: Paths + distribution ────────────────────────────────────────────────
cells.append(code("""\
# --- Kaggle Dataset Paths ---
CSV_PATH = '../input/datasets/swailumzafar01/zohroof-dataset-1/ISIC_2019_Training_GroundTruth.csv'
IMG_DIR  = '../input/datasets/swailumzafar01/zohroof-dataset-1/ISIC_2019_Training_Input/ISIC_2019_Training_Input'

df = pd.read_csv(CSV_PATH)
malignant_cols = ['MEL', 'BCC', 'SCC', 'AK']
df_mal = df[df[malignant_cols].sum(axis=1) > 0].reset_index(drop=True)
print(f'Total malignant images: {len(df_mal)}')

class_counts = {col: int(df_mal[col].sum()) for col in malignant_cols}
total = sum(class_counts.values())
print('\\nClass distribution:')
for cls, cnt in class_counts.items():
    print(f'  {cls}: {cnt} ({100*cnt/total:.1f}%)')

colors_bar = ['#4C72B0', '#55A868', '#C44E52', '#DD8452']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

bars = axes[0].bar(class_counts.keys(), class_counts.values(),
                   color=colors_bar, edgecolor='white', linewidth=1.5)
for bar, (cls, cnt) in zip(bars, class_counts.items()):
    pct = 100 * cnt / total
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                f'{cnt}\\n({pct:.1f}%)', ha='center', va='bottom',
                fontweight='bold', fontsize=11)
axes[0].set_title('Malignant Subtype Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Images', fontsize=12)
axes[0].set_xlabel('Cancer Subtype', fontsize=12)

explode = (0, 0, 0.12, 0.12)
_, _, autotexts = axes[1].pie(
    class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%',
    colors=colors_bar, explode=explode, startangle=90,
    textprops={'fontsize': 12}, pctdistance=0.75)
for at in autotexts:
    at.set_fontweight('bold')
axes[1].set_title('Class Proportion (SCC & AK highlighted)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('class_imbalance_stage2b.png', dpi=150, bbox_inches='tight')
plt.show()

mel_count = class_counts['MEL']
print(f'\\nMEL outnumbers SCC by {mel_count/class_counts["SCC"]:.1f}:1')
""", 'cell_03'))

# ── 4: Section 2 header ───────────────────────────────────────────────────────
cells.append(md("""\
## 2. Dataset, Undersampling & Dual-Transform Support

### Step 1 — Hard Undersample MEL & BCC to 2000

Before any train/val split, we randomly select at most **2000** images from each
majority class. SCC and AK are **untouched** (they are already small).

Result:
| Class | Before | After |
|-------|--------|-------|
| MEL   | 4 522  | **2 000** |
| BCC   | 3 323  | **2 000** |
| SCC   | 628    | 628 |
| AK    | 867    | 867 |
| **Total** | **9 340** | **~6 495** |

### Step 2 — 80/20 Stratified Train/Val Split

### Step 3 — Dual-Transform Datasets
- `train_ds` → rich augmentation
- `val_ds`   → deterministic (Resize 256 → CenterCrop 224)
""", 'cell_04'))

# ── 5: Dataset + transforms + undersampling ─────────────────────────────────
cells.append(code("""\
MAX_MAJORITY = 2000   # hard cap for MEL and BCC
RANDOM_SEED  = 42

class ISIC2019MalignantDataset(Dataset):
    \\\"\\\"\\\"ISIC 2019 dataset filtered to 4 malignant subtypes.
    Supports an optional index mask for undersampling majority classes.
    \\\"\\\"\\\"

    def __init__(self, csv_file, root_dir, transform=None, subset_indices=None):
        full_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.malignant_cols = ['MEL', 'BCC', 'SCC', 'AK']

        # Keep only malignant rows
        mal_df = full_df[full_df[self.malignant_cols].sum(axis=1) > 0].reset_index(drop=True)

        # Apply optional index mask (for undersampling)
        if subset_indices is not None:
            mal_df = mal_df.iloc[subset_indices].reset_index(drop=True)

        self.df = mal_df

        # Pre-compute integer labels
        self.labels = []
        for _, row in self.df.iterrows():
            for i, col in enumerate(self.malignant_cols):
                if row[col] == 1.0:
                    self.labels.append(i)
                    break
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image'] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Step 1: Build the undersampled index list ─────────────────────────────────
# Load full malignant CSV to get per-class indices
full_df_raw    = pd.read_csv(CSV_PATH)
mal_cols       = ['MEL', 'BCC', 'SCC', 'AK']
mal_df_raw     = full_df_raw[full_df_raw[mal_cols].sum(axis=1) > 0].reset_index(drop=True)

rng = np.random.default_rng(RANDOM_SEED)
kept_indices = []
print('Undersampling majority classes to MAX_MAJORITY =', MAX_MAJORITY)
print(f'{"Class":<8} {"Before":>8} {"After":>8}')
print('-' * 28)
for i, col in enumerate(mal_cols):
    cls_idx = mal_df_raw.index[mal_df_raw[col] == 1.0].tolist()
    if len(cls_idx) > MAX_MAJORITY:
        cls_idx = rng.choice(cls_idx, size=MAX_MAJORITY, replace=False).tolist()
        after = MAX_MAJORITY
    else:
        after = len(cls_idx)
    print(f'{col:<8} {len(mal_df_raw[mal_df_raw[col]==1.0]):>8} {after:>8}')
    kept_indices.extend(cls_idx)

kept_indices = sorted(kept_indices)
print(f'{"TOTAL":<8} {len(mal_df_raw):>8} {len(kept_indices):>8}')


# ── Step 2: Augmentation Transforms ─────────────────────────────────────────
# Train: rich augmentation (benefits SCC/AK most since WRS makes them appear often)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

# Val: deterministic, no augmentation
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Step 3: Two dataset objects with the same undersampled index mask ─────────
train_ds_full = ISIC2019MalignantDataset(csv_file=CSV_PATH, root_dir=IMG_DIR,
                                          transform=train_transform,
                                          subset_indices=kept_indices)
val_ds_full   = ISIC2019MalignantDataset(csv_file=CSV_PATH, root_dir=IMG_DIR,
                                          transform=val_transform,
                                          subset_indices=kept_indices)

all_labels = train_ds_full.labels   # shared index space (undersampled)

# ── Step 4: 80/20 stratified split ───────────────────────────────────────────
train_idx, val_idx = train_test_split(
    np.arange(len(train_ds_full)), test_size=0.2,
    stratify=all_labels, random_state=RANDOM_SEED
)

train_subset = Subset(train_ds_full, train_idx)
val_subset   = Subset(val_ds_full,   val_idx)

print(f'\\nTotal after undersampling : {len(train_ds_full)}')
print(f'Training set              : {len(train_subset)}')
print(f'Validation set            : {len(val_subset)}')

train_labels = all_labels[train_idx]
val_labels   = all_labels[val_idx]
print('\\nPer-class counts (Train / Val):')
for i, name in enumerate(CLASS_NAMES):
    tr_cnt = int((train_labels == i).sum())
    va_cnt = int((val_labels == i).sum())
    print(f'  {name}: {tr_cnt} / {va_cnt}')

# ── Visualise before vs after undersampling ───────────────────────────────────
before_counts = {col: int(mal_df_raw[col].sum()) for col in mal_cols}
after_counts  = {col: int((all_labels == i).sum()) for i, col in enumerate(mal_cols)}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_bar = ['#4C72B0', '#55A868', '#C44E52', '#DD8452']

for ax, counts, title in zip(axes,
                              [before_counts, after_counts],
                              ['Before Undersampling (full dataset)',
                               'After Undersampling (MEL/BCC capped at 2 000)']):
    bars = ax.bar(counts.keys(), counts.values(), color=colors_bar,
                  edgecolor='white', linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Image Count')
    ax.axhline(MAX_MAJORITY, color='red', linestyle='--', alpha=0.6,
               label=f'Cap = {MAX_MAJORITY}')
    ax.legend(fontsize=9)
    for j, (col, v) in enumerate(counts.items()):
        ax.text(j, v + 30, str(v), ha='center', fontweight='bold', fontsize=11)

plt.suptitle('Majority-Class Undersampling — Stage 2b', fontsize=14,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('undersampling_stage2b.png', dpi=150, bbox_inches='tight')
plt.show()
""", 'cell_05'))

# ── 6: Section 3 header ───────────────────────────────────────────────────────
cells.append(md("""\
## 3. WeightedRandomSampler — No Naive Oversampling for SCC/AK

We already **undersampled** MEL/BCC. Now we use `WeightedRandomSampler`
to make SCC/AK appear **more often per epoch** without duplicating files.

| Method for minority (SCC/AK) | Risk of overfitting | Notes |
|------------------------------|-------------------|-------|
| Replicate rare images | **High ❌** | Model memorises the same 628 SCC images |
| **WeightedRandomSampler** | **Low ✅** | Fresh weighted draw each epoch — no duplication |

### Result after combining both steps

| Class | Real images | Effective samples/epoch |
|-------|------------|------------------------|
| MEL   | 1 600 (train) | ~balanced via WRS |
| BCC   | 1 600 (train) | ~balanced via WRS |
| SCC   | 502 (train)  | ~balanced via WRS |
| AK    | 694 (train)  | ~balanced via WRS |
""", 'cell_06'))

# ── 7: WeightedRandomSampler ────────────────────────────────────────────────
cells.append(code("""\
# ── WeightedRandomSampler ─────────────────────────────────────────────────────

train_class_counts = Counter(train_labels.tolist())
print('Training split class counts:')
for i, name in enumerate(CLASS_NAMES):
    print(f'  {name}: {train_class_counts[i]}')

# Weight per sample: inverse class frequency
sample_weights = np.array([
    1.0 / train_class_counts[int(label)]
    for label in train_labels
])

# replacement=True is required and intentional for the math to work.
sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

# NOTE: sampler is mutually exclusive with shuffle=True in DataLoader
train_loader = DataLoader(train_subset, batch_size=32, sampler=sampler,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_subset,   batch_size=32, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f'\\nTrain DataLoader: {len(train_loader)} batches/epoch')
print(f'Val   DataLoader: {len(val_loader)} batches/epoch')

# ── Verify sampler balance ─────────────────────────────────────────────────
sample_check = Counter()
for _, lbls in train_loader:
    for l in lbls.numpy():
        sample_check[int(l)] += 1

print('\\nSampler check — effective class distribution in 1 epoch:')
for i, name in enumerate(CLASS_NAMES):
    pct = 100 * sample_check[i] / sum(sample_check.values())
    print(f'  {name}: {sample_check[i]:5d} samples ({pct:.1f}%)')

# ── Visualise original vs sampler ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

orig_counts = [int((train_labels == i).sum()) for i in range(NUM_CLASSES)]
samp_counts = [sample_check[i] for i in range(NUM_CLASSES)]
colors_bar  = ['#4C72B0', '#55A868', '#C44E52', '#DD8452']

for ax, counts, title in zip(axes,
                               [orig_counts, samp_counts],
                               ['Original Training Distribution',
                                'WeightedRandomSampler — Effective per-Epoch']):
    bars = ax.bar(CLASS_NAMES, counts, color=colors_bar, edgecolor='white', linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Image Count' if 'Original' in title else 'Samples Drawn')
    for j, v in enumerate(counts):
        ax.text(j, v + 20, str(v), ha='center', fontweight='bold')

plt.suptitle('Before vs After WeightedRandomSampler', fontsize=14,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('sampler_distribution_stage2b.png', dpi=150, bbox_inches='tight')
plt.show()
""", 'cell_07'))

# ── 8: Section 4 header ───────────────────────────────────────────────────────
cells.append(md("""\
## 4. Model Definitions — ImageNet Pretrained Weights

Unlike Stage 2a (random initialisation), Stage 2b uses **ImageNet pretrained weights**.

Benefits:
- Lower epochs needed (30-50 vs 100+)
- Better feature extraction from day 1
- Less likely to get stuck in poor local minima

Only the **classification head** is swapped to 4 output classes.
The backbone is fine-tuned end-to-end (all layers trainable).
""", 'cell_08'))

# ── 9: Model definitions ──────────────────────────────────────────────────
cells.append(code("""\
def get_stage2b_models(num_classes=4):
    \"\"\"Create 5 models WITH ImageNet pretrained weights (fine-tune all layers).\"\"\"
    models = {
        'ResNet50'       : torchvision.models.resnet50(
                               weights=torchvision.models.ResNet50_Weights.DEFAULT),
        'EfficientNet-B0': torchvision.models.efficientnet_b0(
                               weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT),
        'MobileNetV2'    : torchvision.models.mobilenet_v2(
                               weights=torchvision.models.MobileNet_V2_Weights.DEFAULT),
        'ConvNeXt-Tiny'  : torchvision.models.convnext_tiny(
                               weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT),
        'Swin-Tiny'      : torchvision.models.swin_t(
                               weights=torchvision.models.Swin_T_Weights.DEFAULT),
    }

    # Replace classification heads for 4-class output
    models['ResNet50'].fc = nn.Linear(
        models['ResNet50'].fc.in_features, num_classes)
    models['EfficientNet-B0'].classifier[1] = nn.Linear(
        models['EfficientNet-B0'].classifier[1].in_features, num_classes)
    models['MobileNetV2'].classifier[1] = nn.Linear(
        models['MobileNetV2'].classifier[1].in_features, num_classes)
    models['ConvNeXt-Tiny'].classifier[2] = nn.Linear(
        models['ConvNeXt-Tiny'].classifier[2].in_features, num_classes)
    models['Swin-Tiny'].head = nn.Linear(
        models['Swin-Tiny'].head.in_features, num_classes)

    return models


ensemble_models = get_stage2b_models(num_classes=NUM_CLASSES)
print('Stage 2b models — ImageNet pretrained weights:')

model_names  = list(ensemble_models.keys())
param_counts = [sum(p.numel() for p in m.parameters()) / 1e6
                for m in ensemble_models.values()]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(model_names, param_counts,
               color=['#4C72B0','#55A868','#C44E52','#DD8452','#8172B3'],
               edgecolor='white', linewidth=1.2)
for bar, cnt in zip(bars, param_counts):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{cnt:.1f}M', va='center', fontweight='bold', fontsize=11)
ax.set_xlabel('Parameters (Millions)', fontsize=12)
ax.set_title('Model Complexity — Stage 2b (Pretrained Fine-tune)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('model_params_stage2b.png', dpi=150, bbox_inches='tight')
plt.show()

for name, cnt in zip(model_names, param_counts):
    print(f'  {name:20s}: {cnt:7.2f}M parameters')
""", 'cell_09'))

# ── 10: Section 5 header ──────────────────────────────────────────────────────
cells.append(md("""\
## 5. Loss Function — Class-Weighted Focal Loss (Double Safety Net)

Two complementary mechanisms work together:

1. **WeightedRandomSampler** → balanced mini-batches (batch-level balance)
2. **Class-weighted Focal Loss** → extra gradient signal for rare classes *within* each batch

Class weights follow the inverse-frequency formula (sklearn convention):
```
w_c = total_samples / (num_classes × count_c)
```
This ensures `sum(w_c × count_c) == total_samples`, keeping the total
loss magnitude similar to unweighted loss.
""", 'cell_10'))

# ── 11: FocalLoss + class weights ─────────────────────────────────────────
cells.append(code("""\
# ── Class weights ─────────────────────────────────────────────────────────────
class_counts_train = np.array([int((train_labels == i).sum()) for i in range(NUM_CLASSES)])
total_train        = class_counts_train.sum()
class_weights      = total_train / (NUM_CLASSES * class_counts_train)
class_weights_t    = torch.FloatTensor(class_weights).to(device)

print('Class weights for loss (higher = rarer class):')
for name, w in zip(CLASS_NAMES, class_weights):
    print(f'  {name}: {w:.4f}')


class FocalLoss(nn.Module):
    \"\"\"Focal Loss with optional per-class weights.

    FL(p_t) = -w_t * (1 - p_t)^gamma * log(p_t)

    gamma > 0  → easy examples down-weighted (hard-example mining)
    weight     → rare classes up-weighted via cross-entropy weight
    \"\"\"

    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss    = F.cross_entropy(inputs, targets,
                                     weight=self.weight, reduction='none')
        pt         = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def mixup_data(x, y, alpha=0.4):
    \"\"\"MixUp augmentation. alpha=0.4 gives lighter mixing — better for imbalanced data.\"\"\"
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x.size(0)).to(x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


print('\\nFocal Loss (gamma=2) + class-weighted CE defined.')
print('MixUp (alpha=0.4) defined.')
""", 'cell_11'))

# ── 12: Section 6 header ──────────────────────────────────────────────────────
cells.append(md("""\
## 6. Training Functions

- Max **50 epochs** (pretrained models converge faster than from-scratch)
- **Patience 10** — early stopping on validation loss
- **CosineAnnealingLR** — smooth LR decay, well-suited for fine-tuning
- **MixUp** (alpha=0.4) applied during training only
""", 'cell_12'))

# ── 13: EarlyStopping + training loop ─────────────────────────────────────
cells.append(code("""\
class EarlyStopping:
    \"\"\"Stop when val loss does not improve for `patience` consecutive epochs.\"\"\"

    def __init__(self, patience=10, verbose=True, delta=0.0):
        self.patience  = patience
        self.verbose   = verbose
        self.delta     = delta
        self.best_loss = None
        self.counter   = 0
        self.early_stop      = False
        self.best_model_wts  = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss      = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter        = 0
            if self.verbose:
                print(f'  [ES] Val loss improved to {val_loss:.4f} — weights saved')
        else:
            self.counter += 1
            if self.verbose:
                print(f'  [ES] No improvement ({self.counter}/{self.patience})')
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    \"\"\"One training epoch with MixUp.\"\"\"
    model.train()
    running_loss, n = 0.0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        mixed_x, y_a, y_b, lam = mixup_data(inputs, labels, alpha=0.4)
        optimizer.zero_grad()
        outputs = model(mixed_x)
        loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        n            += inputs.size(0)
    return running_loss / n


def validate(model, dataloader, criterion, device):
    \"\"\"Validation — no MixUp.\"\"\"
    model.eval()
    running_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            correct      += (outputs.argmax(dim=1) == labels).sum().item()
            n            += inputs.size(0)
    return running_loss / n, correct / n


def train_model(model, train_loader, val_loader, criterion,
                optimizer, scheduler, device, num_epochs=50, patience=10):
    \"\"\"Full fine-tuning loop with early stopping + cosine LR.\"\"\"
    es      = EarlyStopping(patience=patience, verbose=True)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, num_epochs + 1):
        t0                  = time.time()
        train_loss          = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc   = validate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch:3d}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e} | '
              f'Time: {elapsed:.1f}s')

        es(val_loss, model)
        if es.early_stop:
            print(f'  Early stopping at epoch {epoch}.')
            break

    model.load_state_dict(es.best_model_wts)
    return model, history


print('Training functions defined: EarlyStopping, train_one_epoch, validate, train_model.')
""", 'cell_13'))

# ── 14: Section 7 header ──────────────────────────────────────────────────────
cells.append(md("""\
## 7. Train All 5 Models

Fine-tune from ImageNet weights for up to **50 epochs** (patience=10).

| Hyper-param | Value | Rationale |
|-------------|-------|-----------|
| Optimizer | AdamW | Weight decay regularises fine-tuning |
| LR | 1e-4 | Lower than from-scratch to avoid destroying pretrained features |
| Weight decay | 1e-4 | Light regularisation |
| Scheduler | CosineAnnealingLR (T_max=50) | Smooth lr decay |
""", 'cell_14'))

# ── 15: Training loop ─────────────────────────────────────────────────────
cells.append(code("""\
NUM_EPOCHS = 50
PATIENCE   = 10
LR         = 1e-4    # lower than scratch training
WD         = 1e-4

all_histories  = {}
trained_models = {}

# Focal Loss (gamma=2) + inverse-frequency class weights
criterion = FocalLoss(gamma=2.0, weight=class_weights_t)

for name, model in ensemble_models.items():
    print(f'\\n{"="*65}')
    print(f'Training {name} (pretrained → fine-tune) ...')
    print(f'{"="*65}')

    model     = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    model, history = train_model(
        model, train_loader, val_loader, criterion,
        optimizer, scheduler, device,
        num_epochs=NUM_EPOCHS, patience=PATIENCE
    )

    trained_models[name] = model
    all_histories[name]  = history
    print(f'{name} → Best Val Loss: {min(history["val_loss"]):.4f} | '
          f'Best Val Acc: {max(history["val_acc"]):.4f}')

print('\\n' + '='*65)
print('ALL 5 MODELS TRAINED SUCCESSFULLY — Stage 2b')
print('='*65)
""", 'cell_15'))

# ── 16: Section 8 ────────────────────────────────────────────────────────────
cells.append(md('## 8. Training Curves\n', 'cell_16'))

# ── 17: Training curves ───────────────────────────────────────────────────
cells.append(code("""\
n_models = len(all_histories)
fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))
colors_train, colors_val = '#4C72B0', '#C44E52'

for i, (name, hist) in enumerate(all_histories.items()):
    epochs = range(1, len(hist['train_loss']) + 1)

    axes[0, i].plot(epochs, hist['train_loss'], label='Train', color=colors_train, linewidth=2)
    axes[0, i].plot(epochs, hist['val_loss'],   label='Val',   color=colors_val,   linewidth=2)
    axes[0, i].set_title(f'{name}\\nLoss', fontsize=12, fontweight='bold')
    axes[0, i].set_xlabel('Epoch')
    axes[0, i].set_ylabel('Loss')
    axes[0, i].legend(fontsize=9)
    axes[0, i].grid(True, alpha=0.3)

    best_ep  = int(np.argmax(hist['val_acc'])) + 1
    best_acc = max(hist['val_acc'])
    axes[1, i].plot(epochs, hist['val_acc'], label='Val Acc', color='#55A868', linewidth=2)
    axes[1, i].axvline(best_ep, color='gray', linestyle='--', alpha=0.7)
    axes[1, i].annotate(f'Best: {best_acc:.3f}\\n(ep {best_ep})',
                        xy=(best_ep, best_acc), fontsize=9,
                        arrowprops=dict(arrowstyle='->', color='gray'),
                        xytext=(best_ep + 1, best_acc - 0.06))
    axes[1, i].set_title(f'{name}\\nVal Accuracy', fontsize=12, fontweight='bold')
    axes[1, i].set_xlabel('Epoch')
    axes[1, i].set_ylabel('Accuracy')
    axes[1, i].legend(fontsize=9)
    axes[1, i].grid(True, alpha=0.3)

plt.suptitle(
    'Stage 2b Training Curves — Pretrained + WeightedRandomSampler + Class-Weighted Focal Loss',
    fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('training_curves_stage2b.png', dpi=150, bbox_inches='tight')
plt.show()
""", 'cell_17'))

# ── 18: Eval header ───────────────────────────────────────────────────────────
cells.append(md("""\
## 9. Model Evaluation
### 9.1 Confusion Matrices
""", 'cell_18'))

# ── 19: Confusion matrices ────────────────────────────────────────────────
cells.append(code("""\
def get_all_predictions(model, dataloader, device):
    \"\"\"Collect predictions, true labels, and softmax probs.\"\"\"
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs  = inputs.to(device)
            outputs = model(inputs)
            probs   = F.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


model_predictions = {}
for name, model in trained_models.items():
    preds, labels, probs = get_all_predictions(model, val_loader, device)
    model_predictions[name] = {'preds': preds, 'labels': labels, 'probs': probs}

fig, axes = plt.subplots(1, 5, figsize=(30, 5))
for ax, (name, data) in zip(axes, model_predictions.items()):
    cm  = confusion_matrix(data['labels'], data['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    acc = accuracy_score(data['labels'], data['preds'])
    ax.set_title(f'{name}\\nAcc: {acc:.3f}', fontsize=12, fontweight='bold')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')

plt.suptitle('Confusion Matrices — Stage 2b (Pretrained + WRS)',
             fontsize=15, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('confusion_matrices_stage2b.png', dpi=150, bbox_inches='tight')
plt.show()
""", 'cell_19'))

# ── 20: ROC header ────────────────────────────────────────────────────────────
cells.append(md("""\
### 9.2 ROC Curves (One-vs-Rest)

ROC is computed per class using the One-vs-Rest strategy.
High AUC for SCC and AK confirms the sampler is working.
""", 'cell_20'))

# ── 21: ROC curves ────────────────────────────────────────────────────────
cells.append(code("""\
fig, axes = plt.subplots(1, 5, figsize=(30, 6))
roc_colors = ['#4C72B0', '#55A868', '#C44E52', '#DD8452']

for ax, (name, data) in zip(axes, model_predictions.items()):
    labels_bin = label_binarize(data['labels'], classes=[0, 1, 2, 3])
    auc_vals   = []
    for cls_idx in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(labels_bin[:, cls_idx], data['probs'][:, cls_idx])
        roc_auc     = auc(fpr, tpr)
        auc_vals.append(roc_auc)
        ax.plot(fpr, tpr, color=roc_colors[cls_idx], linewidth=2,
                label=f'{CLASS_NAMES[cls_idx]} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_title(f'{name}\\nMacro AUC: {np.mean(auc_vals):.3f}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('ROC Curves (One-vs-Rest) — Stage 2b',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('roc_curves_stage2b.png', dpi=150, bbox_inches='tight')
plt.show()
""", 'cell_21'))

# ── 22: Per-class header ──────────────────────────────────────────────────────
cells.append(md("""\
### 9.3 Per-Class Performance — Focus on Rare Cancers (SCC, AK)

**Key clinical question**: Did WeightedRandomSampler improve SCC and AK recall
compared to Stage 2a?

Red borders highlight the rare-class bars.
""", 'cell_22'))

# ── 23: Per-class recall ──────────────────────────────────────────────────
cells.append(code("""\
per_class_recall = {name: [] for name in model_predictions}

for name, data in model_predictions.items():
    print(f'\\n{"="*60}')
    print(f'{name} — Classification Report')
    print('='*60)
    report = classification_report(data['labels'], data['preds'],
                                   target_names=CLASS_NAMES, digits=4)
    print(report)
    report_dict = classification_report(data['labels'], data['preds'],
                                        target_names=CLASS_NAMES, output_dict=True)
    for cls_name in CLASS_NAMES:
        per_class_recall[name].append(report_dict[cls_name]['recall'])

fig, ax = plt.subplots(figsize=(14, 6))
x     = np.arange(len(CLASS_NAMES))
width = 0.15
model_colors = ['#4C72B0', '#55A868', '#C44E52', '#DD8452', '#8172B3']

for i, (name, recalls) in enumerate(per_class_recall.items()):
    offset = (i - 2) * width
    bars = ax.bar(x + offset, recalls, width, label=name,
                  color=model_colors[i], edgecolor='white', linewidth=0.8)
    for j in [2, 3]:   # SCC, AK
        bars[j].set_edgecolor('#C44E52')
        bars[j].set_linewidth(2.5)

ax.set_xlabel('Cancer Subtype', fontsize=13)
ax.set_ylabel('Recall (Sensitivity)', fontsize=13)
ax.set_title('Per-Class Recall — Did WeightedRandomSampler Help Rare Cancers? (Stage 2b)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(0, 1.05)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('per_class_recall_stage2b.png', dpi=150, bbox_inches='tight')
plt.show()
""", 'cell_23'))

# ── 24: PR curves header ──────────────────────────────────────────────────────
cells.append(md("""\
### 9.4 Precision-Recall Curves

PR curves are more informative than ROC for imbalanced datasets.
A high PR-AUC for SCC/AK confirms the sampler is effective.
""", 'cell_24'))

# ── 25: PR curves ─────────────────────────────────────────────────────────
cells.append(code("""\
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
model_colors_pr = ['#4C72B0', '#55A868', '#C44E52', '#DD8452', '#8172B3']

true_lbl_ref = list(model_predictions.values())[0]['labels']
labels_bin_ref = label_binarize(true_lbl_ref, classes=[0, 1, 2, 3])

for cls_idx, (ax, cls_name) in enumerate(zip(axes, CLASS_NAMES)):
    for (name, data), color in zip(model_predictions.items(), model_colors_pr):
        prec, rec, _ = precision_recall_curve(
            labels_bin_ref[:, cls_idx], data['probs'][:, cls_idx])
        ap = average_precision_score(labels_bin_ref[:, cls_idx], data['probs'][:, cls_idx])
        ax.plot(rec, prec, color=color, linewidth=2, label=f'{name} (AP={ap:.3f})')

    baseline = labels_bin_ref[:, cls_idx].mean()
    ax.axhline(baseline, color='k', linestyle='--', alpha=0.4,
               label=f'Baseline ({baseline:.3f})')
    ax.set_title(f'{cls_name} — Precision-Recall', fontsize=13, fontweight='bold')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

plt.suptitle('Precision-Recall Curves — Stage 2b',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('pr_curves_stage2b.png', dpi=150, bbox_inches='tight')
plt.show()
""", 'cell_25'))

# ── 26: Ensemble header ───────────────────────────────────────────────────────
cells.append(md("""\
## 10. Ensemble Evaluation

Average softmax probabilities across all 5 models → argmax for final prediction.
""", 'cell_26'))

# ── 27: Ensemble eval ─────────────────────────────────────────────────────
cells.append(code("""\
all_probs_list = [data['probs'] for data in model_predictions.values()]
ensemble_probs = np.mean(all_probs_list, axis=0)
ensemble_preds = np.argmax(ensemble_probs, axis=1)
true_labels    = list(model_predictions.values())[0]['labels']

fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# 1) Confusion matrix
cm = confusion_matrix(true_labels, ensemble_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
ens_acc = accuracy_score(true_labels, ensemble_preds)
axes[0].set_title(f'Ensemble Confusion Matrix\\nAcc: {ens_acc:.3f}',
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel('True')
axes[0].set_xlabel('Predicted')

# 2) Ensemble OvR ROC
labels_bin = label_binarize(true_labels, classes=[0, 1, 2, 3])
roc_colors = ['#4C72B0', '#55A868', '#C44E52', '#DD8452']
auc_vals_e = []
for cls_idx in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(labels_bin[:, cls_idx], ensemble_probs[:, cls_idx])
    roc_auc = auc(fpr, tpr)
    auc_vals_e.append(roc_auc)
    axes[1].plot(fpr, tpr, color=roc_colors[cls_idx], linewidth=2,
                 label=f'{CLASS_NAMES[cls_idx]} (AUC={roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[1].set_title(f'Ensemble OvR ROC\\nMacro AUC: {np.mean(auc_vals_e):.3f}',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('FPR')
axes[1].set_ylabel('TPR')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# 3) Recall comparison: best individual vs ensemble
ens_report  = classification_report(true_labels, ensemble_preds,
                                     target_names=CLASS_NAMES, output_dict=True)
ens_recalls = [ens_report[c]['recall'] for c in CLASS_NAMES]

best_name, best_f1 = None, 0
for name, data in model_predictions.items():
    rep = classification_report(data['labels'], data['preds'],
                                target_names=CLASS_NAMES, output_dict=True)
    f1  = rep['macro avg']['f1-score']
    if f1 > best_f1:
        best_f1, best_name = f1, name

best_report  = classification_report(
    model_predictions[best_name]['labels'],
    model_predictions[best_name]['preds'],
    target_names=CLASS_NAMES, output_dict=True)
best_recalls = [best_report[c]['recall'] for c in CLASS_NAMES]

x = np.arange(len(CLASS_NAMES))
axes[2].bar(x - 0.2, best_recalls, 0.35, label=f'Best: {best_name}',
            color='#4C72B0', edgecolor='white')
axes[2].bar(x + 0.2, ens_recalls,  0.35, label='Ensemble',
            color='#AB63FA', edgecolor='white')
axes[2].set_xticks(x)
axes[2].set_xticklabels(CLASS_NAMES, fontsize=12)
axes[2].set_ylabel('Recall')
axes[2].set_ylim(0, 1.05)
axes[2].set_title('Recall: Best Individual vs Ensemble', fontsize=13, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3, axis='y')

plt.suptitle('Ensemble Evaluation — Stage 2b', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ensemble_evaluation_stage2b.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Ensemble Accuracy : {ens_acc:.4f}')
print(f'Ensemble Macro F1 : {ens_report["macro avg"]["f1-score"]:.4f}')
print(f'Ensemble Macro AUC: {np.mean(auc_vals_e):.4f}')
""", 'cell_27'))

# ── 28: Radar header ──────────────────────────────────────────────────────────
cells.append(md("""\
## 11. Radar Chart & Summary Table

6 key metrics across all 5 models + ensemble.
""", 'cell_28'))

# ── 29: Radar + summary ───────────────────────────────────────────────────
cells.append(code("""\
import matplotlib.patches as mpatches

metric_names    = ['Overall Acc', 'MEL Recall', 'BCC Recall',
                   'SCC Recall', 'AK Recall', 'Macro F1']
n_metrics       = len(metric_names)
all_model_metrics = {}

for name, data in model_predictions.items():
    rep = classification_report(data['labels'], data['preds'],
                                target_names=CLASS_NAMES, output_dict=True)
    acc = accuracy_score(data['labels'], data['preds'])
    all_model_metrics[name] = [
        acc,
        rep['MEL']['recall'], rep['BCC']['recall'],
        rep['SCC']['recall'], rep['AK']['recall'],
        rep['macro avg']['f1-score'],
    ]

all_model_metrics['Ensemble'] = [
    ens_acc,
    ens_report['MEL']['recall'], ens_report['BCC']['recall'],
    ens_report['SCC']['recall'], ens_report['AK']['recall'],
    ens_report['macro avg']['f1-score'],
]

# Radar chart
angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]
radar_colors = ['#4C72B0','#55A868','#C44E52','#DD8452','#8172B3','#AB63FA']

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, (name, vals) in enumerate(all_model_metrics.items()):
    values = vals + vals[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=name, color=radar_colors[i])
    ax.fill(angles, values, alpha=0.05, color=radar_colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_names, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('Clinical Suitability Radar — Stage 2b',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig('radar_chart_stage2b.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary table
print('\\n' + '='*85)
print(f'{"Model":<20} {"Acc":>8} {"MEL-R":>8} {"BCC-R":>8} {"SCC-R":>8} {"AK-R":>8} {"MacroF1":>9}')
print('-'*85)
for name, vals in all_model_metrics.items():
    marker = ' *' if name == 'Ensemble' else ''
    print(f'{name:<20} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} '
          f'{vals[3]:>8.4f} {vals[4]:>8.4f} {vals[5]:>9.4f}{marker}')
print('='*85)
print('* = Ensemble (average softmax)')
""", 'cell_29'))

# ── 30: Save header ───────────────────────────────────────────────────────────
cells.append(md("""\
## 12. Save Model Weights
""", 'cell_30'))

# ── 31: Save weights ──────────────────────────────────────────────────────
cells.append(code("""\
for name, model in trained_models.items():
    safe_name = name.replace('/', '_').replace(' ', '_').lower()
    path = f'{safe_name}_stage2b.pth'
    torch.save(model.state_dict(), path)
    print(f'Saved: {path}')

print('\\n' + '='*65)
print('STAGE 2b COMPLETE')
print('='*65)
print('Approach  : ImageNet pretrained + WeightedRandomSampler')
print('Loss      : Focal Loss (gamma=2) + inverse-frequency class weights')
print('Augment   : RandomResizedCrop, Flips, Rotation, ColorJitter, RandomErasing')
print('Epochs    : up to 50 with patience=10 early stopping')
print(f'Ensemble Acc    : {ens_acc:.4f}')
print(f'Ensemble Macro F1: {ens_report["macro avg"]["f1-score"]:.4f}')
print(f'Ensemble Macro AUC: {np.mean(auc_vals_e):.4f}')
""", 'cell_31'))

# ─────────────────────────────────────────────────────────────────────────────
# Write notebook
# ─────────────────────────────────────────────────────────────────────────────
notebook = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.11.0'
        }
    },
    'cells': cells
}

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Written  : {OUT_PATH}')
print(f'Cells    : {len(cells)}')
print(f'Size     : {os.path.getsize(OUT_PATH):,} bytes')
