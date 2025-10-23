# ===========================================
# prediction.py — Consistent with TB_RFUnet_P.py
# Validation uses per-pixel labels; outputs val metrics + full-image map
# ===========================================
import os
import random
import yaml
import torch
import numpy as np
import rasterio
from torch.utils.data import DataLoader
from sklearn.utils import check_array
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier

from models.dual_branch_unet import DualBranchUNet
from data.dataloader import HyperspectralDataset, load_data
from utils.spectral_features import calculate_spectral_relationship_features
from utils.feature_engineer import HSIFeatureEngineer

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)



def _concat_npy(files):
    arrs = [np.load(p) for p in files]
    return np.concatenate(arrs, axis=0) if len(arrs) > 0 else np.empty((0,), dtype=np.float32)


def run_prediction(config):
    # ---- Configs
    paths = config.get('paths', {})
    model_cfg = config.get('model', {})
    train_cfg = config.get('train', {})

    hsi_path = paths.get('hsi_data')
    msi_path = paths.get('msi_data')
    depth_path = paths.get('depth_data')
    label_path = paths.get('label_data')
    output_dir = paths.get('output_dir', 'outputs/')
    os.makedirs(output_dir, exist_ok=True)

    patch_size = int(train_cfg.get('patch_size', 32))
    stride_size = int(train_cfg.get('stride_size', 24))   # match baseline split
    batch_size = int(train_cfg.get('batch_size', 16))
    # force feature_dim=512 to match baseline
    feature_dim = 64

    # ---- Load data
    print("Loading dataset ...")
    hsi_image, msi_image, depth_feature, labels, profile = load_data(
        hsi_path, depth_path, msi_path, label_path
    )
    num_classes = int(np.max(labels) + 1) if 0 in np.unique(labels) else len(np.unique(labels))
    print(f"Dataset loaded. Shapes: HSI{hsi_image.shape} MSI{msi_image.shape} DEP{depth_feature.shape} LBL{labels.shape}")
    print(f"Number of classes: {num_classes}")

    # ---- Load model
    model_path = os.path.join(output_dir, "tbunet_hsi_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchUNet(
        hsi_channels=hsi_image.shape[-1],
        msi_channels=msi_image.shape[-1],
        num_classes=num_classes,
        feature_dim=feature_dim,     # force 512
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model from: {model_path}")

    # ---- Create full dataset and 50/50 split (identical to baseline)
    print("Creating full dataset and splitting into train/validation ...")
    full_dataset = HyperspectralDataset(
        hsi_image, msi_image, depth_feature, labels,
        patch_size=patch_size, stride=stride_size
    )
    np.random.seed(42)
    num_train = int(len(full_dataset) * 0.5)
    train_indices = np.random.choice(len(full_dataset), size=num_train, replace=False)
    val_indices = np.setdiff1d(np.arange(len(full_dataset)), train_indices)

    train_patches = full_dataset.patches[train_indices]
    train_msi_patches = full_dataset.msi_patches[train_indices]
    train_depth_patches = full_dataset.depth_patches[train_indices]
    train_patch_labels = full_dataset.patch_labels[train_indices]
    train_indices_list = [full_dataset.patch_indices[i] for i in train_indices]

    val_patches = full_dataset.patches[val_indices]
    val_msi_patches = full_dataset.msi_patches[val_indices]
    val_depth_patches = full_dataset.depth_patches[val_indices]
    val_patch_labels = full_dataset.patch_labels[val_indices]
    val_indices_list = [full_dataset.patch_indices[i] for i in val_indices]

    print(f"Selected {len(train_indices)} patches for training, {len(val_indices)} for validation")

    # ---- Build datasets/dataloaders (shuffle=False for determinism)
    train_dataset = HyperspectralDataset(hsi_image, msi_image, depth_feature, labels,
                                         patch_size=patch_size, stride=stride_size)
    train_dataset.patches = train_patches
    train_dataset.msi_patches = train_msi_patches
    train_dataset.depth_patches = train_depth_patches
    train_dataset.patch_labels = train_patch_labels
    train_dataset.patch_indices = train_indices_list
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    val_dataset = HyperspectralDataset(hsi_image, msi_image, depth_feature, labels,
                                       patch_size=patch_size, stride=stride_size)
    val_dataset.patches = val_patches
    val_dataset.msi_patches = val_msi_patches
    val_dataset.depth_patches = val_depth_patches
    val_dataset.patch_labels = val_patch_labels
    val_dataset.patch_indices = val_indices_list
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # ---- Feature engineer
    engineer = HSIFeatureEngineer(model, num_classes=num_classes, patch_size=patch_size)

    # ---------- TRAIN: extract (per-pixel) to disk ----------
    print("Extracting training features ...")
    tr_feat_files, tr_depth_files, tr_label_files, _ = engineer.extract_features(train_loader, is_train=True)
    train_deep = _concat_npy(tr_feat_files).astype(np.float32)
    train_depth = _concat_npy(tr_depth_files).astype(np.float32)
    train_labels = _concat_npy(tr_label_files).astype(np.int64)

    # RAW per-pixel (flatten in same order as deep/depth)
    H, W = patch_size, patch_size
    train_raw = train_dataset.patches.reshape(-1, train_dataset.patches.shape[-1]).astype(np.float32)

    # SPECTRAL per-pixel (compute on patches, then flatten like above)
    spec_train = calculate_spectral_relationship_features(train_dataset.patches).astype(np.float32)
    # If spectral is per-patch (Npatch,F), expand to per-pixel:
    n_pix = train_deep.shape[0]
    if spec_train.shape[0] == len(train_dataset.patches):
        spec_train = np.repeat(spec_train, H * W, axis=0)

    # Consistency checks
    assert train_depth.shape[0] == n_pix
    assert train_raw.shape[0] == n_pix
    assert spec_train.shape[0] == n_pix
    assert train_labels.shape[0] == n_pix

    # Combine + impute (train uses fit_transform)
    train_combined = engineer.prepare_combined_features(train_deep, train_raw, train_depth, spec_train)
    train_combined = check_array(train_combined, ensure_all_finite="allow-nan").astype(np.float32)
    train_combined[np.isinf(train_combined)] = np.nan
    imputer_train = SimpleImputer(strategy="mean")
    train_combined = imputer_train.fit_transform(train_combined).astype(np.float32)

    # ---------- VAL: extract (per-pixel) to disk ----------
    print("Extracting validation features ...")
    val_feat_files, val_depth_files, val_label_files, _ = engineer.extract_features(val_loader, is_train=False)
    val_deep = _concat_npy(val_feat_files).astype(np.float32)
    val_depth = _concat_npy(val_depth_files).astype(np.float32)
    if len(val_label_files) > 0:
        val_labels_px = _concat_npy(val_label_files).astype(np.int64)
    else:
        # unlikely since val has labels; fallback to center-only then repeat
        val_labels_px = np.repeat(val_dataset.patch_labels.reshape(-1), H * W).astype(np.int64)

    val_raw = val_dataset.patches.reshape(-1, val_dataset.patches.shape[-1]).astype(np.float32)
    spec_val = calculate_spectral_relationship_features(val_dataset.patches).astype(np.float32)
    n_val_pix = val_deep.shape[0]
    if spec_val.shape[0] == len(val_dataset.patches):
        spec_val = np.repeat(spec_val, H * W, axis=0)

    assert val_depth.shape[0] == n_val_pix
    assert val_raw.shape[0] == n_val_pix
    assert spec_val.shape[0] == n_val_pix
    assert val_labels_px.shape[0] == n_val_pix

    val_combined = engineer.prepare_combined_features(val_deep, val_raw, val_depth, spec_val)
    val_combined = check_array(val_combined, ensure_all_finite="allow-nan").astype(np.float32)
    val_combined[np.isinf(val_combined)] = np.nan


    # val_combined = imputer_train.transform(val_combined)
    batch = 100000
    N = val_combined.shape[0]
    buf = np.empty_like(val_combined, dtype=val_combined.dtype)
    for s in range(0, N, batch):
        e = min(s + batch, N)
        buf[s:e] = imputer_train.transform(val_combined[s:e])
    val_combined = buf


    # ---- RF train
    print("Training Random Forest classifier ...")
    # ensure params identical (engineer already has), but set explicitly
    engineer.rf_classifier = RandomForestClassifier(
        n_estimators=100, max_depth=16, max_features="sqrt",
        n_jobs=-1, class_weight="balanced", random_state=42
    )
    engineer.rf_classifier.fit(train_combined, train_labels)

    # ---- Evaluate on validation set
    print("Evaluating on validation set ...")
    val_predictions = engineer.predict(val_combined)
    classes = list(range(num_classes))
    report = classification_report(val_labels_px, val_predictions, labels=classes, output_dict=True)
    oa = report["accuracy"]
    cls_prec = [report[str(i)]["precision"] for i in classes if str(i) in report]
    aa = float(np.mean(cls_prec)) if len(cls_prec) > 0 else 0.0
    kappa = cohen_kappa_score(val_labels_px, val_predictions)

    print("\n==== Validation Results ====")
    print(f"Overall Accuracy (OA): {oa:.6f}")
    print(f"Average Accuracy (AA): {aa:.6f}")
    print(f"Kappa Coefficient: {kappa:.6f}")
    for i, acc in enumerate(cls_prec):
        print(f"Class {i} precision: {acc:.6f}")

    # ---- Full image prediction (non-overlap stride = patch_size)
    print("\nRunning full image prediction ...")
    full_dataset_all = HyperspectralDataset(hsi_image, msi_image, depth_feature, labels=None,
                                            patch_size=patch_size, stride=patch_size)
    full_loader = DataLoader(full_dataset_all, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    full_feat_files, full_depth_files, _, full_patch_indices = engineer.extract_features(full_loader, is_train=False)
    full_deep = _concat_npy(full_feat_files).astype(np.float32)
    full_depth = _concat_npy(full_depth_files).astype(np.float32)
    full_raw = full_dataset_all.patches.reshape(-1, full_dataset_all.patches.shape[-1]).astype(np.float32)
    full_spec = calculate_spectral_relationship_features(full_dataset_all.patches).astype(np.float32)
    n_full = full_deep.shape[0]
    if full_spec.shape[0] == len(full_dataset_all.patches):
        full_spec = np.repeat(full_spec, H * W, axis=0)

    assert full_depth.shape[0] == n_full
    assert full_raw.shape[0] == n_full
    assert full_spec.shape[0] == n_full

    full_combined = engineer.prepare_combined_features(full_deep, full_raw, full_depth, full_spec)
    full_combined = check_array(full_combined, ensure_all_finite="allow-nan").astype(np.float32)
    full_combined[np.isinf(full_combined)] = np.nan

    preds = engineer.predict_in_batches(full_combined, batch_size=200000)
    segmentation = engineer.reconstruct_segmentation(preds, full_patch_indices, hsi_image.shape[:2])

    # ---- Save result
    out_path = os.path.join(output_dir, "rf_prediction_segmentation_50split.tif")
    profile.update(dtype=rasterio.int64, count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(segmentation.astype(rasterio.int64), 1)
    print(f"Saved full segmentation result to: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_prediction(config)