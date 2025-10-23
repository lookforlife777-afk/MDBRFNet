import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, cohen_kappa_score

from models.dual_branch_unet import DualBranchUNet
from data.dataloader import HyperspectralDataset, load_data


def evaluate_model(model, dataloader, num_classes):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for hsi_patches, msi_patches, depth_patches, patch_labels, _ in tqdm(dataloader, desc="Evaluating"):
            hsi_patches = hsi_patches.to(device).float()
            msi_patches = msi_patches.to(device).float()
            depth_patches = depth_patches.to(device).float()
            patch_labels = patch_labels.to(device).long()

            _, logits = model(hsi_patches, msi_patches, depth_patches)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(patch_labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    # Mask out invalid (e.g., 0 label) pixels if needed
    mask = all_labels > 0
    preds_valid = all_preds[mask]
    labels_valid = all_labels[mask]

    # Compute confusion matrix
    cm = confusion_matrix(labels_valid, preds_valid, labels=np.arange(1, num_classes))
    per_class_acc = np.diag(cm) / cm.sum(axis=1)
    OA = np.sum(np.diag(cm)) / np.sum(cm)
    AA = np.mean(per_class_acc)
    kappa = cohen_kappa_score(labels_valid, preds_valid)

    print("\n===== Evaluation Results =====")
    for i, acc in enumerate(per_class_acc, start=1):
        print(f"Class {i} Accuracy: {acc * 100:.2f}%")
    print(f"\nOverall Accuracy (OA): {OA * 100:.2f}%")
    print(f"Average Accuracy (AA): {AA * 100:.2f}%")
    print(f"Kappa Coefficient: {kappa:.4f}")

    return per_class_acc, OA, AA, kappa


def main():
    # === Config ===
    model_path = "outputs/tbunet_hsi_model_e100.pth"
    hsi_path = "data/hyperspectral.tif"
    msi_path = "data/multispectral.tif"
    depth_path = "data/depth.tif"
    label_path = "data/label.tif"
    patch_size = 32
    batch_size = 16
    feature_dim = 64

    # === Load Data ===
    hsi_image, msi_image, depth_feature, labels, _ = load_data(hsi_path, depth_path, msi_path, label_path)
    num_classes = len(np.unique(labels))

    dataset = HyperspectralDataset(hsi_image, msi_image, depth_feature, labels,
                                   patch_size=patch_size, stride=patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # === Load Model ===
    model = DualBranchUNet(hsi_channels=hsi_image.shape[-1],
                           msi_channels=msi_image.shape[-1],
                           num_classes=num_classes,
                           feature_dim=feature_dim)

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"✅ Loaded trained model from {model_path}")

    # === Evaluate ===
    evaluate_model(model, dataloader, num_classes)


if __name__ == "__main__":
    main()
