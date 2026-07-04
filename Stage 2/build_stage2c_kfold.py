"""
build_stage2c_kfold.py
Generates Stage 2c: 5-Fold Cross-Validation Diverse Ensemble

Architecture:
  - Undersample MEL/BCC to 2000 (same as Stage 2b)
  - StratifiedKFold(5) on the reduced dataset
  - Each fold trains ONE unique architecture (maximum ensemble diversity)
    Fold 0 -> ResNet50
    Fold 1 -> EfficientNet-B0
    Fold 2 -> MobileNetV2
    Fold 3 -> ConvNeXt-Tiny
    Fold 4 -> Swin-Tiny
  - Each model's val fold = the 20% it NEVER saw during training
  - Final prediction = average softmax across all 5 models
  - WeightedRandomSampler + Class-Weighted Focal Loss (carried from 2b)
"""

import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

OUT_PATH = os.path.join(os.path.dirname(__file__), 'stage-2c-kfold-ensemble.ipynb')


def code(src, cell_id):
    return {
        'cell_type': 'code', 'id': cell_id, 'metadata': {},
        'source': src if isinstance(src, list) else [src],
        'outputs': [], 'execution_count': None
    }


def md(src, cell_id):
    return {
        'cell_type': 'markdown', 'id': cell_id, 'metadata': {},
        'source': src if isinstance(src, list) else [src]
    }


cells = []

# ── 0: Title ──────────────────────────────────────────────────────────────────
cells.append(md("""\
# Stage 2c: 5-Fold Cross-Validation Diverse Ensemble
**ISIC 2019 | Pretrained Weights | Undersample MEL/BCC=2000 | K-Fold Diverse Ensemble**

## Key Idea

Instead of training all 5 models on the **same** 80/20 split, we assign each model
to a **different fold**. This gives the ensemble maximum diversity — each model has
seen a completely different 80% of the data during training.

```
Undersampled dataset (~6 495 images)
         │
  StratifiedKFold(n_splits=5)
         │
  ┌──────┼──────┬──────┬──────┐
 Fold0  Fold1  Fold2  Fold3  Fold4   ← each is a unique 20% val slice
  │      │      │      │      │
ResNet  Effic  MobNet ConvNX  Swin   ← 1 arch per fold
  │      │      │      │      │
  └──────┴──────┴──────┴──────┘
              ↓
    Average softmax probabilities
              ↓
         Final prediction
```

## Why this is better than Stage 2b

| | Stage 2b (80/20 fixed) | Stage 2c (K-Fold Ensemble) |
|---|---|---|
| Ensemble diversity | Low — all trained on same 80% | **High** — each model sees different data |
| Val coverage | 20% fixed slice | **Every image** is val exactly once |
| Memorisation risk | Medium | **Lower** (different training sets) |
| Training time | Same | **Same** (5 models × 50 epochs) |
| Thesis robustness | Single split | **Cross-validated** — publishable standard |
""", 'cell_00'))

# ── 1: Imports ─────────────────────────────────────────────────────────────────
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
from sklearn.model_selection import StratifiedKFold
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

CLASS_NAMES  = ['MEL', 'BCC', 'SCC', 'AK']
NUM_CLASSES  = 4
N_FOLDS      = 5
MAX_MAJORITY = 2000   # hard cap MEL and BCC
RANDOM_SEED  = 42
""", 'cell_01'))

# ── 2: Section 1 ──────────────────────────────────────────────────────────────
cells.append(md("""\
## 1. Dataset Paths & Class Distribution
""", 'cell_02'))

# ── 3: Paths + distribution ───────────────────────────────────────────────────
cells.append(code("""\
CSV_PATH = '../input/datasets/swailumzafar01/zohroof-dataset-1/ISIC_2019_Training_GroundTruth.csv'
IMG_DIR  = '../input/datasets/swailumzafar01/zohroof-dataset-1/ISIC_2019_Training_Input/ISIC_2019_Training_Input'

df = pd.read_csv(CSV_PATH)
mal_cols = ['MEL', 'BCC', 'SCC', 'AK']
df_mal   = df[df[mal_cols].sum(axis=1) > 0].reset_index(drop=True)
print(f'Total malignant images: {len(df_mal)}')

class_counts = {col: int(df_mal[col].sum()) for col in mal_cols}
total = sum(class_counts.values())
print('\\nClass distribution:')
for cls, cnt in class_counts.items():
    print(f'  {cls}: {cnt} ({100*cnt/total:.1f}%)')

colors_bar = ['#4C72B0', '#55A868', '#C44E52', '#DD8452']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bars = axes[0].bar(class_counts.keys(), class_counts.values(),
                   color=colors_bar, edgecolor='white', linewidth=1.5)
for bar, (cls, cnt) in zip(bars, class_counts.items()):
    pct = 100 * cnt / total
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                f'{cnt}\\n({pct:.1f}%)', ha='center', va='bottom',
                fontweight='bold', fontsize=11)
axes[0].set_title('Original Malignant Subtype Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Images')

explode = (0, 0, 0.12, 0.12)
_, _, autotexts = axes[1].pie(
    class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%',
    colors=colors_bar, explode=explode, startangle=90,
    textprops={'fontsize': 12}, pctdistance=0.75)
for at in autotexts:
    at.set_fontweight('bold')
axes[1].set_title('Class Proportions (SCC & AK highlighted)', fontsize=13, fontweight='bold')

plt.suptitle('Stage 2c — Class Imbalance', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('class_imbalance_stage2c.png', dpi=150, bbox_inches='tight')
plt.show()
""", 'cell_03'))

# ── 4: Section 2 ──────────────────────────────────────────────────────────────
cells.append(md("""\
## 2. Undersample MEL/BCC → Build Master Index

Hard-cap MEL and BCC to **2000** each (random draw without replacement, seed=42).
SCC and AK are untouched. This produces the **master index list** shared
across all 5 folds — the undersampling happens once, globally, before any split.

> **Note on ordering:** Undersampling before splitting is valid here because we
> are *removing* majority samples, not synthesising minority ones.
> There is no data leakage: the val fold contains only real images the
> training model never saw.
""", 'cell_04'))

# ── 5: Undersampling ──────────────────────────────────────────────────────────
cells.append(code("""\
rng = np.random.default_rng(RANDOM_SEED)
kept_indices = []

print(f'{"Class":<8} {"Before":>8} {"After":>8}')
print('-' * 28)
for col in mal_cols:
    cls_idx = df_mal.index[df_mal[col] == 1.0].tolist()
    before  = len(cls_idx)
    if before > MAX_MAJORITY:
        cls_idx = rng.choice(cls_idx, size=MAX_MAJORITY, replace=False).tolist()
        after = MAX_MAJORITY
    else:
        after = before
    print(f'{col:<8} {before:>8} {after:>8}')
    kept_indices.extend(cls_idx)

kept_indices = sorted(kept_indices)
print(f'{"TOTAL":<8} {len(df_mal):>8} {len(kept_indices):>8}')

# Build the reduced dataframe and label array
df_reduced    = df_mal.iloc[kept_indices].reset_index(drop=True)
all_labels    = np.array([
    next(i for i, col in enumerate(mal_cols) if row[col] == 1.0)
    for _, row in df_reduced.iterrows()
])

print(f'\\nReduced dataset: {len(df_reduced)} images')
print('Per-class:')
for i, cls in enumerate(CLASS_NAMES):
    print(f'  {cls}: {int((all_labels == i).sum())}')

# Visualise
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
before_c = {col: int(df_mal[col].sum()) for col in mal_cols}
after_c  = {cls: int((all_labels == i).sum()) for i, cls in enumerate(CLASS_NAMES)}

for ax, counts, title in zip(axes,
                              [before_c, after_c],
                              ['Before Undersampling',
                               f'After Undersampling (MEL/BCC ≤ {MAX_MAJORITY})']):
    bars = ax.bar(counts.keys(), counts.values(), color=colors_bar,
                  edgecolor='white', linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Image Count')
    ax.axhline(MAX_MAJORITY, color='red', linestyle='--', alpha=0.6,
               label=f'Cap = {MAX_MAJORITY}')
    ax.legend(fontsize=9)
    for j, (col, v) in enumerate(counts.items()):
        ax.text(j, v + 30, str(v), ha='center', fontweight='bold', fontsize=11)

plt.suptitle('Majority-Class Undersampling — Stage 2c', fontsize=14,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('undersampling_stage2c.png', dpi=150, bbox_inches='tight')
plt.show()
""", 'cell_05'))

# ── 6: Section 3 ──────────────────────────────────────────────────────────────
cells.append(md("""\
## 3. Transforms & Dataset Class
""", 'cell_06'))

# ── 7: Transforms + Dataset ───────────────────────────────────────────────────
cells.append(code("""\
# Train: rich augmentation
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


class ISIC2019MalignantDataset(Dataset):
    \\\"\\\"\\\"Malignant subset of ISIC 2019.
    Accepts a pre-built DataFrame (already filtered + undersampled)
    and a pre-computed integer label array.
    \\\"\\\"\\\"

    def __init__(self, df, labels, root_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.labels    = np.array(labels)
        self.root_dir  = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image'] + '.jpg')
        image    = Image.open(img_path).convert('RGB')
        label    = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label


print('Transforms and Dataset class defined.')
print(f'Train transform: {len(train_transform.transforms)} steps')
print(f'Val   transform: {len(val_transform.transforms)} steps')
""", 'cell_07'))

# ── 8: Section 4 ──────────────────────────────────────────────────────────────
cells.append(md("""\
## 4. Model Definitions — ImageNet Pretrained

Each fold gets ONE architecture (different from all other folds):

| Fold | Architecture | Params |
|------|-------------|--------|
| 0 | ResNet50 | ~25M |
| 1 | EfficientNet-B0 | ~5M |
| 2 | MobileNetV2 | ~3M |
| 3 | ConvNeXt-Tiny | ~28M |
| 4 | Swin-Tiny | ~28M |
""", 'cell_08'))

# ── 9: Model factory ──────────────────────────────────────────────────────────
cells.append(code("""\
# Map fold index → architecture name
FOLD_ARCH = {
    0: 'ResNet50',
    1: 'EfficientNet-B0',
    2: 'MobileNetV2',
    3: 'ConvNeXt-Tiny',
    4: 'Swin-Tiny',
}


def build_model(arch_name, num_classes=4):
    \\\"\\\"\\\"Build ONE pretrained model and replace its head.\\\"\\\"\\\"
    if arch_name == 'ResNet50':
        m = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch_name == 'EfficientNet-B0':
        m = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    elif arch_name == 'MobileNetV2':
        m = torchvision.models.mobilenet_v2(
                weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    elif arch_name == 'ConvNeXt-Tiny':
        m = torchvision.models.convnext_tiny(
                weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)

    elif arch_name == 'Swin-Tiny':
        m = torchvision.models.swin_t(
                weights=torchvision.models.Swin_T_Weights.DEFAULT)
        m.head = nn.Linear(m.head.in_features, num_classes)

    else:
        raise ValueError(f'Unknown architecture: {arch_name}')

    return m


# Quick sanity check
print('Architecture → fold assignment:')
for fold_id, arch in FOLD_ARCH.items():
    m = build_model(arch, num_classes=NUM_CLASSES)
    params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f'  Fold {fold_id}: {arch:<20s} {params:6.1f}M params')
    del m
""", 'cell_09'))

# ── 10: Section 5 ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 5. Loss, Training Functions & Early Stopping
""", 'cell_10'))

# ── 11: Loss + training ───────────────────────────────────────────────────────
cells.append(code("""\
class FocalLoss(nn.Module):
    \\\"\\\"\\\"Focal Loss with optional per-class weights.
    FL(p_t) = -w_t * (1-p_t)^gamma * log(p_t)
    \\\"\\\"\\\"
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma     = gamma
        self.weight    = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce   = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class EarlyStopping:
    def __init__(self, patience=10, verbose=True):
        self.patience        = patience
        self.verbose         = verbose
        self.best_loss       = None
        self.counter         = 0
        self.early_stop      = False
        self.best_model_wts  = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss      = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter        = 0
            if self.verbose:
                print(f'  [ES] ✓ Val loss improved to {val_loss:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'  [ES] No improvement ({self.counter}/{self.patience})')
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mx, ya, yb, lam = mixup_data(x, y, alpha=0.4)
        optimizer.zero_grad()
        loss = mixup_criterion(criterion, model(mx), ya, yb, lam)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n          += x.size(0)
    return total_loss / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y    = x.to(device), y.to(device)
            out     = model(x)
            loss    = criterion(out, y)
            probs   = F.softmax(out, dim=1)
            preds   = out.argmax(dim=1)
            total_loss += loss.item() * x.size(0)
            correct    += (preds == y).sum().item()
            n          += x.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (total_loss / n, correct / n,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def train_fold(model, train_loader, val_loader, criterion, device,
               num_epochs=50, patience=10, lr=1e-4, wd=1e-4):
    \\\"\\\"\\\"Train one fold with CosineAnnealingLR + early stopping.\\\"\\\"\\\"
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    es        = EarlyStopping(patience=patience, verbose=True)
    history   = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, num_epochs + 1):
        t0                        = time.time()
        train_loss                = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'  Ep {epoch:3d}/{num_epochs} | '
              f'Train={train_loss:.4f} | Val={val_loss:.4f} | '
              f'Acc={val_acc:.4f} | LR={optimizer.param_groups[0]["lr"]:.1e} | '
              f'{elapsed:.1f}s')

        es(val_loss, model)
        if es.early_stop:
            print(f'  Early stop at epoch {epoch}.')
            break

    model.load_state_dict(es.best_model_wts)
    return model, history


print('Loss, training functions, and EarlyStopping defined.')
""", 'cell_11'))

# ── 12: Section 6 ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 6. 5-Fold Cross-Validation Training Loop

`StratifiedKFold` ensures each fold has the same class proportions.

For each fold:
1. Split the master index into train/val
2. Compute class weights on **that fold's training set** (no leakage)
3. Build `WeightedRandomSampler` on training indices only
4. Train the assigned architecture
5. Collect predictions on the val fold
""", 'cell_12'))

# ── 13: Main CV loop ──────────────────────────────────────────────────────────
cells.append(code("""\
NUM_EPOCHS = 50
PATIENCE   = 10
LR         = 1e-4
WD         = 1e-4

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

fold_results  = {}   # fold_id -> {'model', 'history', 'preds', 'labels', 'probs', 'arch'}

for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
    arch_name = FOLD_ARCH[fold_id]
    print(f'\\n{"="*70}')
    print(f'FOLD {fold_id}  |  Architecture: {arch_name}')
    print(f'  Train: {len(train_idx)} samples  |  Val: {len(val_idx)} samples')
    print(f'{"="*70}')

    # ── Subset DataFrames ────────────────────────────────────────────────────
    train_df  = df_reduced.iloc[train_idx]
    val_df    = df_reduced.iloc[val_idx]
    train_lbl = all_labels[train_idx]
    val_lbl   = all_labels[val_idx]

    print('  Per-class (train / val):')
    for i, cls in enumerate(CLASS_NAMES):
        print(f'    {cls}: {int((train_lbl==i).sum())} / {int((val_lbl==i).sum())}')

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = ISIC2019MalignantDataset(train_df, train_lbl, IMG_DIR, train_transform)
    val_ds   = ISIC2019MalignantDataset(val_df,   val_lbl,   IMG_DIR, val_transform)

    # ── WeightedRandomSampler (computed on this fold's train set only) ───────
    fold_class_counts = Counter(train_lbl.tolist())
    sample_weights    = np.array([1.0 / fold_class_counts[int(l)] for l in train_lbl])
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Class-weighted Focal Loss (computed on this fold's train set only) ───
    fold_counts  = np.array([int((train_lbl == i).sum()) for i in range(NUM_CLASSES)])
    fold_weights = train_lbl.shape[0] / (NUM_CLASSES * fold_counts)
    criterion    = FocalLoss(gamma=2.0,
                             weight=torch.FloatTensor(fold_weights).to(device))
    print(f'  Class weights: ' +
          ', '.join(f'{n}={w:.3f}' for n, w in zip(CLASS_NAMES, fold_weights)))

    # ── Build & train model ──────────────────────────────────────────────────
    model = build_model(arch_name, num_classes=NUM_CLASSES).to(device)

    model, history = train_fold(
        model, train_loader, val_loader, criterion, device,
        num_epochs=NUM_EPOCHS, patience=PATIENCE, lr=LR, wd=WD
    )

    # ── Collect val predictions ──────────────────────────────────────────────
    _, val_acc, preds, true_labels, probs = validate(model, val_loader, criterion, device)
    print(f'  Fold {fold_id} Best Val Acc: {val_acc:.4f}')

    fold_results[fold_id] = {
        'arch'   : arch_name,
        'model'  : model,
        'history': history,
        'preds'  : preds,
        'labels' : true_labels,
        'probs'  : probs,
        'val_idx': val_idx,
    }

print('\\n' + '='*70)
print('ALL 5 FOLDS TRAINED SUCCESSFULLY — Stage 2c')
print('='*70)
""", 'cell_13'))

# ── 14: Section 7 ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 7. Training Curves — All Folds
""", 'cell_14'))

# ── 15: Training curves ───────────────────────────────────────────────────────
cells.append(code("""\
fig, axes = plt.subplots(2, N_FOLDS, figsize=(5 * N_FOLDS, 10))
fold_colors = ['#4C72B0','#55A868','#C44E52','#DD8452','#8172B3']

for fold_id, res in fold_results.items():
    hist = res['history']
    arch = res['arch']
    eps  = range(1, len(hist['train_loss']) + 1)
    col  = fold_colors[fold_id]

    axes[0, fold_id].plot(eps, hist['train_loss'], label='Train', color=col, linewidth=2)
    axes[0, fold_id].plot(eps, hist['val_loss'],   label='Val',   color=col, linewidth=2, linestyle='--')
    axes[0, fold_id].set_title(f'Fold {fold_id}: {arch}\\nLoss', fontsize=11, fontweight='bold')
    axes[0, fold_id].set_xlabel('Epoch')
    axes[0, fold_id].set_ylabel('Loss')
    axes[0, fold_id].legend(fontsize=8)
    axes[0, fold_id].grid(True, alpha=0.3)

    best_ep  = int(np.argmax(hist['val_acc'])) + 1
    best_acc = max(hist['val_acc'])
    axes[1, fold_id].plot(eps, hist['val_acc'], color=col, linewidth=2, label='Val Acc')
    axes[1, fold_id].axvline(best_ep, color='gray', linestyle='--', alpha=0.7)
    axes[1, fold_id].annotate(f'{best_acc:.3f}\\nep{best_ep}',
                               xy=(best_ep, best_acc), fontsize=9,
                               arrowprops=dict(arrowstyle='->', color='gray'),
                               xytext=(best_ep + 1, best_acc - 0.07))
    axes[1, fold_id].set_title(f'Fold {fold_id}: {arch}\\nVal Acc', fontsize=11, fontweight='bold')
    axes[1, fold_id].set_xlabel('Epoch')
    axes[1, fold_id].set_ylabel('Accuracy')
    axes[1, fold_id].legend(fontsize=8)
    axes[1, fold_id].grid(True, alpha=0.3)

plt.suptitle('Stage 2c — Training Curves per Fold (K-Fold Diverse Ensemble)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('training_curves_stage2c.png', dpi=150, bbox_inches='tight')
plt.show()
""", 'cell_15'))

# ── 16: Section 8 ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 8. Per-Fold Evaluation

Each fold's model is evaluated on its own val set (the 20% it never saw).
Since the folds are disjoint, together they cover the **full reduced dataset**.
""", 'cell_16'))

# ── 17: Per-fold metrics ──────────────────────────────────────────────────────
cells.append(code("""\
fig, axes = plt.subplots(1, N_FOLDS, figsize=(28, 5))

print('Per-Fold Classification Reports')
print('='*70)

per_fold_metrics = {}
for fold_id, res in fold_results.items():
    preds  = res['preds']
    labels = res['labels']
    arch   = res['arch']

    acc = accuracy_score(labels, preds)
    rep = classification_report(labels, preds, target_names=CLASS_NAMES,
                                output_dict=True, digits=4)
    per_fold_metrics[fold_id] = {'acc': acc, 'report': rep, 'arch': arch}

    print(f'\\nFold {fold_id} ({arch}) — Val Acc: {acc:.4f}')
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[fold_id],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[fold_id].set_title(f'Fold {fold_id}: {arch}\\nAcc={acc:.3f}',
                             fontsize=11, fontweight='bold')
    axes[fold_id].set_ylabel('True')
    axes[fold_id].set_xlabel('Predicted')

plt.suptitle('Confusion Matrices — Stage 2c K-Fold (each model on its own val fold)',
             fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('confusion_matrices_stage2c.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary row
print('\\n' + '='*80)
print(f'{"Fold":<6} {"Arch":<20} {"Acc":>7} {"MEL-R":>8} {"BCC-R":>8} {"SCC-R":>8} {"AK-R":>8} {"MacroF1":>9}')
print('-'*80)
for fid, m in per_fold_metrics.items():
    r = m['report']
    print(f'{fid:<6} {m["arch"]:<20} {m["acc"]:>7.4f} '
          f'{r["MEL"]["recall"]:>8.4f} {r["BCC"]["recall"]:>8.4f} '
          f'{r["SCC"]["recall"]:>8.4f} {r["AK"]["recall"]:>8.4f} '
          f'{r["macro avg"]["f1-score"]:>9.4f}')
print('='*80)
""", 'cell_17'))

# ── 18: Section 9 ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 9. Ensemble Prediction (Average Softmax Across All 5 Folds)

Each model predicts on **all images** (not just its val fold) to form the ensemble.
This is the key step: after training, each model is applied to the full dataset,
and their softmax scores are averaged.
""", 'cell_18'))

# ── 19: Full ensemble predictions ─────────────────────────────────────────────
cells.append(code("""\
# ── Each model predicts on its own val fold.
# For the ENSEMBLE test, we collect predictions on the FULL reduced dataset.
# Build a complete-dataset loader (val transform, no shuffle)
full_ds     = ISIC2019MalignantDataset(df_reduced, all_labels, IMG_DIR, val_transform)
full_loader = DataLoader(full_ds, batch_size=32, shuffle=False,
                         num_workers=2, pin_memory=True)


def predict_full(model, loader, device):
    \\\"\\\"\\\"Run inference on the full dataset, return probs.\\\"\\\"\\\"
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out   = model(x.to(device))
            probs = F.softmax(out, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    return np.array(all_probs), np.array(all_labels)


print('Running full-dataset inference for each fold model ...')
all_fold_probs = []
for fold_id, res in fold_results.items():
    probs, true_lbl = predict_full(res['model'], full_loader, device)
    all_fold_probs.append(probs)
    print(f'  Fold {fold_id} ({res["arch"]}) done.')

# Average softmax
ensemble_probs = np.mean(all_fold_probs, axis=0)
ensemble_preds = np.argmax(ensemble_probs, axis=1)

ens_acc = accuracy_score(true_lbl, ensemble_preds)
ens_rep = classification_report(true_lbl, ensemble_preds,
                                 target_names=CLASS_NAMES, output_dict=True)

print(f'\\nEnsemble Accuracy : {ens_acc:.4f}')
print(f'Ensemble Macro F1 : {ens_rep["macro avg"]["f1-score"]:.4f}')
print()
print(classification_report(true_lbl, ensemble_preds, target_names=CLASS_NAMES, digits=4))
""", 'cell_19'))

# ── 20: Section 10 ────────────────────────────────────────────────────────────
cells.append(md("""\
## 10. Ensemble Evaluation — Confusion Matrix, ROC, Recall
""", 'cell_20'))

# ── 21: Ensemble eval plots ───────────────────────────────────────────────────
cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# 1) Confusion matrix
cm = confusion_matrix(true_lbl, ensemble_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[0].set_title(f'Ensemble Confusion Matrix\\nAcc: {ens_acc:.3f}',
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel('True')
axes[0].set_xlabel('Predicted')

# 2) Ensemble OvR ROC
labels_bin = label_binarize(true_lbl, classes=[0, 1, 2, 3])
roc_colors = ['#4C72B0', '#55A868', '#C44E52', '#DD8452']
auc_vals   = []
for cls_idx in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(labels_bin[:, cls_idx], ensemble_probs[:, cls_idx])
    roc_auc     = auc(fpr, tpr)
    auc_vals.append(roc_auc)
    axes[1].plot(fpr, tpr, color=roc_colors[cls_idx], linewidth=2,
                 label=f'{CLASS_NAMES[cls_idx]} (AUC={roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[1].set_title(f'Ensemble OvR ROC\\nMacro AUC: {np.mean(auc_vals):.3f}',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('FPR')
axes[1].set_ylabel('TPR')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# 3) Per-class recall bar chart
ens_recalls  = [ens_rep[c]['recall'] for c in CLASS_NAMES]
fold_recalls  = {f'Fold{fid} ({r["arch"][:6]})':
                 [r['report'][c]['recall'] for c in CLASS_NAMES]
                 for fid, r in per_fold_metrics.items()}

x     = np.arange(NUM_CLASSES)
width = 0.12
fold_pal = ['#4C72B0','#55A868','#C44E52','#DD8452','#8172B3']
for i, (name, rec) in enumerate(fold_recalls.items()):
    axes[2].bar(x + (i-2)*width, rec, width, label=name, color=fold_pal[i], alpha=0.6)
axes[2].bar(x + 2.5*width, ens_recalls, width*1.2, label='Ensemble',
            color='#AB63FA', edgecolor='white', linewidth=1.5)
axes[2].set_xticks(x)
axes[2].set_xticklabels(CLASS_NAMES, fontsize=12)
axes[2].set_ylabel('Recall')
axes[2].set_ylim(0, 1.1)
axes[2].set_title('Per-Class Recall: Folds vs Ensemble\\n(red border = rare class)',
                  fontsize=12, fontweight='bold')
axes[2].legend(fontsize=7, loc='upper right')
axes[2].grid(True, alpha=0.3, axis='y')
for j in [2, 3]:   # SCC, AK borders
    for patch in axes[2].patches:
        if abs(patch.get_x() - (j - 0.3)) < 0.6:
            patch.set_edgecolor('#C44E52')

plt.suptitle('Ensemble Evaluation — Stage 2c K-Fold Diverse Ensemble',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ensemble_evaluation_stage2c.png', dpi=150, bbox_inches='tight')
plt.show()
""", 'cell_21'))

# ── 22: Section 11 — Radar ────────────────────────────────────────────────────
cells.append(md("""\
## 11. Radar Chart & Final Summary Table
""", 'cell_22'))

# ── 23: Radar ─────────────────────────────────────────────────────────────────
cells.append(code("""\
metric_names = ['Overall Acc', 'MEL Recall', 'BCC Recall',
                'SCC Recall',  'AK Recall',  'Macro F1']
n_metrics    = len(metric_names)

all_metrics = {}

# Individual fold models (on their own val slice)
for fid, m in per_fold_metrics.items():
    r = m['report']
    all_metrics[f'Fold{fid}:{m["arch"][:5]}'] = [
        m['acc'],
        r['MEL']['recall'], r['BCC']['recall'],
        r['SCC']['recall'], r['AK']['recall'],
        r['macro avg']['f1-score'],
    ]

# Ensemble (full dataset)
all_metrics['Ensemble'] = [
    ens_acc,
    ens_rep['MEL']['recall'], ens_rep['BCC']['recall'],
    ens_rep['SCC']['recall'], ens_rep['AK']['recall'],
    ens_rep['macro avg']['f1-score'],
]

angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]
colors = ['#4C72B0','#55A868','#C44E52','#DD8452','#8172B3','#AB63FA']

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
for i, (name, vals) in enumerate(all_metrics.items()):
    v = vals + vals[:1]
    lw = 3 if 'Ensemble' in name else 1.5
    ax.plot(angles, v, 'o-', linewidth=lw, label=name, color=colors[i % len(colors)])
    ax.fill(angles, v, alpha=0.04, color=colors[i % len(colors)])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_names, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('Clinical Suitability Radar — Stage 2c (K-Fold Diverse Ensemble)',
             fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=8)
plt.tight_layout()
plt.savefig('radar_chart_stage2c.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary table
print('\\n' + '='*90)
print(f'{"Model":<28} {"Acc":>7} {"MEL-R":>8} {"BCC-R":>8} {"SCC-R":>8} {"AK-R":>8} {"MacroF1":>9}')
print('-'*90)
for name, vals in all_metrics.items():
    marker = ' ★' if 'Ensemble' in name else ''
    print(f'{name:<28} {vals[0]:>7.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} '
          f'{vals[3]:>8.4f} {vals[4]:>8.4f} {vals[5]:>9.4f}{marker}')
print('='*90)
print('★ = K-Fold Diverse Ensemble')
""", 'cell_23'))

# ── 24: Section 12 — Save ─────────────────────────────────────────────────────
cells.append(md("""\
## 12. Save Model Weights
""", 'cell_24'))

# ── 25: Save ──────────────────────────────────────────────────────────────────
cells.append(code("""\
for fold_id, res in fold_results.items():
    arch      = res['arch']
    safe_name = arch.replace('/', '_').replace(' ', '_').lower()
    path      = f'{safe_name}_fold{fold_id}_stage2c.pth'
    torch.save(res['model'].state_dict(), path)
    print(f'Saved: {path}')

print('\\n' + '='*70)
print('STAGE 2c — K-FOLD DIVERSE ENSEMBLE COMPLETE')
print('='*70)
print(f'Folds            : {N_FOLDS}')
print(f'Architectures    : {list(FOLD_ARCH.values())}')
print(f'Undersampling    : MEL/BCC capped at {MAX_MAJORITY}')
print(f'Sampler          : WeightedRandomSampler (per fold)')
print(f'Loss             : Focal (gamma=2) + class-weighted CE (per fold)')
print(f'Epochs           : up to {NUM_EPOCHS} with patience={PATIENCE}')
print(f'Ensemble Acc     : {ens_acc:.4f}')
print(f'Ensemble MacroF1 : {ens_rep["macro avg"]["f1-score"]:.4f}')
print(f'Ensemble MacroAUC: {np.mean(auc_vals):.4f}')
""", 'cell_25'))

# ── Write notebook ─────────────────────────────────────────────────────────────
notebook = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11.0'}
    },
    'cells': cells
}

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Written : {OUT_PATH}')
print(f'Cells   : {len(cells)}')
print(f'Size    : {os.path.getsize(OUT_PATH):,} bytes')
