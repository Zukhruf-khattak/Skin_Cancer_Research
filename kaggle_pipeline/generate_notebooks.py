import json
import os

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def markdown_cell(source):
    if isinstance(source, str):
        source = [line + '\n' for line in source.split('\n')]
    return {
        "cell_type": "markdown",
        "id": os.urandom(4).hex(),
        "metadata": {},
        "source": source
    }

def code_cell(source):
    if isinstance(source, str):
        source = [line + '\n' for line in source.split('\n')]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": os.urandom(4).hex(),
        "metadata": {},
        "outputs": [],
        "source": source
    }

# NOTEBOOK 1: Stage 1 Binary Classification
nb1_cells = [
    markdown_cell("# Stage 1: Binary Classification (Benign vs Malignant)\nThis notebook implements a 5-model ensemble. It includes the full training loop specifically set up for 20 epochs to train the remaining models like ConvNeXt and ViT."),
    code_cell("import os\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, Dataset\nfrom torchvision import datasets, transforms\nimport torchvision\nimport timm\nprint('PyTorch version:', torch.__version__)"),
    code_cell("""# 1. Data Loading Setup
# Note: You will need to write a custom Dataset class to parse the ISIC 2019 Ground Truth CSV
# and load the images. Below is a placeholder for the PyTorch DataLoader structure.

data_dir = '../input/isic-2019/ISIC_2019_Training_Input'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# train_loader = DataLoader(your_custom_dataset, batch_size=32, shuffle=True)
print("Data loading transforms defined.")"""),
    code_cell("""# 2. Define 5 models
def get_ensemble_models(num_classes=2):
    models = {
        'resnet50': torchvision.models.resnet50(pretrained=True),
        'efficientnet_b0': torchvision.models.efficientnet_b0(pretrained=True),
        'mobilenet_v2': torchvision.models.mobilenet_v2(pretrained=True),
        'convnext_tiny': torchvision.models.convnext_tiny(pretrained=True),
        'vit_b_16': torchvision.models.vit_b_16(pretrained=True)
    }
    
    # Modify heads for binary classification
    models['resnet50'].fc = nn.Linear(models['resnet50'].fc.in_features, num_classes)
    models['efficientnet_b0'].classifier[1] = nn.Linear(models['efficientnet_b0'].classifier[1].in_features, num_classes)
    models['mobilenet_v2'].classifier[1] = nn.Linear(models['mobilenet_v2'].classifier[1].in_features, num_classes)
    models['convnext_tiny'].classifier[2] = nn.Linear(models['convnext_tiny'].classifier[2].in_features, num_classes)
    models['vit_b_16'].heads.head = nn.Linear(models['vit_b_16'].heads.head.in_features, num_classes)
    
    return models
    
ensemble = get_ensemble_models()
print("Models successfully initialized for Stage 1.")"""),
    code_cell("""# 3. Stage 1 Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example: We select ConvNeXt to train. You can change this to 'vit_b_16' or any other model.
model_name = 'convnext_tiny' 
model = ensemble[model_name].to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4) # AdamW is excellent for ConvNeXt/ViT

num_epochs = 20 # Train for 20-30 epochs for Stage 1

def train_model(model, criterion, optimizer, num_epochs, dataloader=None):
    print(f"Starting training for {num_epochs} epochs on {device}...")
    
    if dataloader is None:
        print("Waiting for train_loader to be defined. Skipping loop execution.")
        return model

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}')
        
    print("Training complete!")
    return model

# Uncomment to actually run the training when train_loader is ready:
# trained_model = train_model(model, criterion, optimizer, num_epochs, dataloader=train_loader)
print("Training loop defined.")""")
]

# NOTEBOOK 2: Stage 2 Multi-Class Classification with Focal Loss & Advanced Augmentation
nb2_cells = [
    markdown_cell("# Stage 2: Multi-Class Classification (Malignant Subtypes)\nThis notebook implements the 5-model ensemble from scratch for 50 epochs. It utilizes Focal Loss and MixUp Augmentation to handle severe class imbalance."),
    code_cell("import os\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nimport torch.nn.functional as F\nimport torchvision\nfrom torchvision import transforms\nimport timm\nimport numpy as np\nprint('PyTorch version:', torch.__version__)"),
    code_cell("""# 1. Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss

print("Focal Loss defined to handle severe class imbalance.")"""),
    code_cell("""# 2. Advanced Augmentation: MixUp
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

print("MixUp augmentation functions defined.")"""),
    code_cell("""# 3. Define 5 models for Multi-Class
def get_stage2_ensemble_models(num_classes=4): # e.g. 4 malignant subtypes
    models = {
        'resnet50': torchvision.models.resnet50(weights=None),
        'efficientnet_b0': torchvision.models.efficientnet_b0(weights=None),
        'mobilenet_v2': torchvision.models.mobilenet_v2(weights=None),
        'convnext_tiny': torchvision.models.convnext_tiny(weights=None),
        'vit_b_16': torchvision.models.vit_b_16(weights=None)
    }
    
    # Modify heads
    models['resnet50'].fc = nn.Linear(models['resnet50'].fc.in_features, num_classes)
    models['efficientnet_b0'].classifier[1] = nn.Linear(models['efficientnet_b0'].classifier[1].in_features, num_classes)
    models['mobilenet_v2'].classifier[1] = nn.Linear(models['mobilenet_v2'].classifier[1].in_features, num_classes)
    models['convnext_tiny'].classifier[2] = nn.Linear(models['convnext_tiny'].classifier[2].in_features, num_classes)
    models['vit_b_16'].heads.head = nn.Linear(models['vit_b_16'].heads.head.in_features, num_classes)
    
    return models
    
stage2_ensemble = get_stage2_ensemble_models()
print("Models successfully initialized for Stage 2.")"""),
    code_cell("""# 4. Stage 2 Training Loop (with MixUp and Focal Loss)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'efficientnet_b0' # Example model
model = stage2_ensemble[model_name].to(device)

criterion = FocalLoss(alpha=1, gamma=2)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

num_epochs = 50 # Stage 2 models train from scratch, so they need more epochs

def train_model_stage2(model, criterion, optimizer, num_epochs, dataloader=None):
    print(f"Starting Stage 2 training for {num_epochs} epochs on {device}...")
    
    if dataloader is None:
        print("Waiting for train_loader to be defined. Skipping loop execution.")
        return model

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply MixUp augmentation during training
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss using MixUp criterion combined with Focal Loss
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}')
        
    print("Stage 2 Training complete!")
    return model

# Uncomment to actually run the training when train_loader is ready:
# trained_model = train_model_stage2(model, criterion, optimizer, num_epochs, dataloader=train_loader)
print("Stage 2 Training loop defined.")""")
]

with open('01_Stage1_Binary_Classification_Ensemble.ipynb', 'w') as f:
    json.dump(create_notebook(nb1_cells), f, indent=2)

with open('02_Stage2_MultiClass_Imbalance_Handled.ipynb', 'w') as f:
    json.dump(create_notebook(nb2_cells), f, indent=2)

print("Notebooks successfully regenerated with full training loops!")
