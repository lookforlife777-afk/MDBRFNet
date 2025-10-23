import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from utils.logger import get_logger
from data.dataloader import HyperspectralDataset, load_data
from models.dual_branch_unet import DualBranchUNet
torch.manual_seed(42)
np.random.seed(42)

def train_unet(unet: DualBranchUNet, train_loader: DataLoader, num_epochs: int = 10, lr: float = 5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet.to(device)
    unet.train()

    # 原脚本里使用了一个类权重向量，这里保留（可根据任务调整）
    class_weight = torch.from_numpy(np.array([0.05, 1.1, 1.6, 1.0, 1.2])).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_seg_acc = 0.0
        total = 0
        mask_cor = 0
        mask_total = 0

        for patches, msi_patches, depth_patches, patch_labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            patches, msi_patches, depth_patches, patch_labels = (
                patches.to(device).float(), msi_patches.to(device).float(),
                depth_patches.to(device).float(), patch_labels.to(device).long())

            optimizer.zero_grad()
            features, seg_logits = unet(patches, msi_patches, depth_patches)
            batch_size, num_classes, h, w = seg_logits.shape
            seg_loss = criterion(seg_logits, patch_labels.view(batch_size, h, w))
            seg_loss.backward()
            optimizer.step()

            running_loss += seg_loss.item()
            _, predicted = seg_logits.max(1)
            total += patch_labels.numel()
            running_seg_acc += predicted.eq(patch_labels).sum().item()

            mask = patch_labels != 0
            pre_mask = predicted[mask]
            labels_mask = patch_labels[mask]
            mask_cor += (pre_mask == labels_mask).sum().item()
            mask_total += labels_mask.numel()

        epoch_loss = running_loss / len(train_loader)
        epoch_seg_acc = 100. * running_seg_acc / total
        eooch_seg_mask_acc = 100. * mask_cor / mask_total if mask_total > 0 else 0.0
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Segmentation Acc: {epoch_seg_acc:.2f}%'
              f', Mask Acc: {eooch_seg_mask_acc:.2f}%')


def train_model(config: dict):
    # load config values
    paths = config.get('paths', {})
    train_cfg = config.get('train', {})
    model_cfg = config.get('model', {})

    hsi_path = paths.get('hsi_data')
    msi_path = paths.get('msi_data')
    depth_path = paths.get('depth_data')
    label_path = paths.get('label_data')
    output_dir = paths.get('output_dir', 'outputs/')

    patch_size = train_cfg.get('patch_size', 32)
    batch_size = train_cfg.get('batch_size', 16)
    epochs = train_cfg.get('epochs', 100)
    lr = train_cfg.get('lr', 5e-4)

    feature_dim = model_cfg.get('feature_dim', 64)

    os.makedirs(output_dir, exist_ok=True)

    hsi_image, msi_image, depth_feature, labels, profile = load_data(hsi_path, depth_path, msi_path, label_path)
    assert hsi_image.shape[:2] == msi_image.shape[:2] == depth_feature.shape[:2], "数据尺寸不匹配"

    full_dataset = HyperspectralDataset(hsi_image, msi_image, depth_feature, labels, patch_size=patch_size, stride=patch_size)

    np.random.seed(42)
    num_train = int(len(full_dataset) * 0.5)
    train_indices = np.random.choice(len(full_dataset), size=num_train, replace=False)

    train_patches = full_dataset.patches[train_indices]
    train_msi_patches = full_dataset.msi_patches[train_indices]
    train_depth_patches = full_dataset.depth_patches[train_indices]
    train_labels = full_dataset.patch_labels[train_indices]
    train_indices_list = [full_dataset.patch_indices[i] for i in train_indices]

    train_dataset = HyperspectralDataset(hsi_image, msi_image, depth_feature, labels, patch_size=patch_size, stride=patch_size)
    train_dataset.patches = train_patches
    train_dataset.msi_patches = train_msi_patches
    train_dataset.depth_patches = train_depth_patches
    train_dataset.patch_labels = train_labels
    train_dataset.patch_indices = train_indices_list

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(np.unique(labels))
    unet = DualBranchUNet(hsi_channels=hsi_image.shape[-1], msi_channels=msi_image.shape[-1], num_classes=num_classes, feature_dim=feature_dim)

    train_unet(unet, train_dataloader, num_epochs=epochs, lr=lr)

    model_path = os.path.join(output_dir, 'tbunet_hsi_model.pth')
    torch.save(unet.state_dict(), model_path)
    print(f"model saved {model_path}")
