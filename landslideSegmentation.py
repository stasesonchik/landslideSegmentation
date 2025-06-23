import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# --- Dataset ---
class LandslideDataset(Dataset):
    def __init__(self, df, folder_path, transform=None):
        self.df = df.reset_index(drop=True)
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def load_and_normalize(self, image_id):
        img_path = os.path.join(self.folder_path, f"{image_id}.npy")
        img = np.load(img_path).astype(np.float32)
        # Normalize each channel independently to [0, 1]
        min_val = img.min(axis=(0, 1), keepdims=True)
        max_val = img.max(axis=(0, 1), keepdims=True)
        img_norm = (img - min_val) / (max_val - min_val + 1e-5)
        return img_norm

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, 'ID']
        img = self.load_and_normalize(image_id)
        # Change shape from (H, W, C) to (C, H, W) for PyTorch
        img = np.transpose(img, (2, 0, 1))
        label = self.df.loc[idx, 'label'] if 'label' in self.df.columns else -1

        if self.transform:
            # Transforms expect PIL Images or tensors; convert numpy to tensor here
            img = torch.from_numpy(img)
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)

# --- Model ---
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 128),  # Assuming input size 64x64, after 4 poolings: 64/2/2/2/2=4
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.clamp(min=1e-7, max=1-1e-7)
        BCE_loss = F.binary_cross_entropy(inputs, targets.unsqueeze(1), reduction='none')
        pt = torch.where(targets.unsqueeze(1) == 1, inputs, 1 - inputs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Metrics ---
def precision_recall_f1(y_true, y_pred, threshold=0.5):
    y_pred_label = (y_pred > threshold).float()
    tp = (y_pred_label * y_true).sum()
    fp = (y_pred_label * (1 - y_true)).sum()
    fn = ((1 - y_pred_label) * y_true).sum()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return precision.item(), recall.item(), f1.item()

# --- Training and Validation functions ---
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    y_true_all = []
    y_pred_all = []

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)  # shape [batch]

        optimizer.zero_grad()
        outputs = model(imgs)            # shape [batch, 1]
        loss = criterion(outputs, labels)  # оба внутри forward будут согласованы
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        y_true_all.append(labels)
        y_pred_all.append(outputs.squeeze().detach())

    epoch_loss = running_loss / len(dataloader.dataset)
    y_true_all = torch.cat(y_true_all)
    y_pred_all = torch.cat(y_pred_all)
    precision, recall, f1 = precision_recall_f1(y_true_all, y_pred_all)
    return epoch_loss, precision, recall, f1


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)  # shape [batch]

            outputs = model(imgs)            # shape [batch, 1]
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            y_true_all.append(labels)
            y_pred_all.append(outputs.squeeze())

    epoch_loss = running_loss / len(dataloader.dataset)
    y_true_all = torch.cat(y_true_all)
    y_pred_all = torch.cat(y_pred_all)
    precision, recall, f1 = precision_recall_f1(y_true_all, y_pred_all)
    return epoch_loss, precision, recall, f1

# --- Main ---

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = 'C:/Users/LeMeS/Desktop/pac/'
    # Paths
    train_csv_path = path + 'Train.csv'
    test_csv_path = path + 'Test.csv'
    train_data_path = path + 'train_data'
    test_data_path = path + 'test_data'

    # Load train dataframe
    train_df = pd.read_csv(train_csv_path, sep = ';')

    # Split train and validation stratified
    train_df_, val_df_ = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)

    # Define transforms (simple tensor conversion here, can add augmentation)
    train_transform = transforms.Compose([
        # Add augmentations here if wanted, e.g. RandomHorizontalFlip(), etc.
        # torchvision.transforms expect PIL Images or tensors; our input already tensor from numpy
    ])

    val_transform = transforms.Compose([])

    # Create datasets and loaders
    train_dataset = LandslideDataset(train_df_, train_data_path, transform=train_transform)
    val_dataset = LandslideDataset(val_df_, train_data_path, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Model, loss, optimizer
    model = CNNModel().to(device)
    criterion = FocalLoss(gamma=2.0, alpha=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop with checkpointing
    best_val_loss = float('inf')
    for epoch in range(50):
        train_loss, train_prec, train_rec, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_prec, val_rec, val_f1 = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1:02d}:")
        print(f"  Train loss: {train_loss:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   loss: {val_loss:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("  Best model saved!")

    # Load best model for inference
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Load test dataframe
    test_df = pd.read_csv(test_csv_path)
    test_dataset = LandslideDataset(test_df, test_data_path, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Prediction on test
    all_preds = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = (outputs.squeeze() > 0.5).long()
            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds).numpy()
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'label': all_preds
    })
    submission.to_csv('Submission_File.csv', index=False)
    print("Submission file saved as 'Submission_File.csv'.")

if __name__ == '__main__':
    main()
