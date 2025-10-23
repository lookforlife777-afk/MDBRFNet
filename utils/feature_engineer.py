import os
import numpy as np
import torch
from typing import Tuple, List
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
torch.manual_seed(42)
np.random.seed(42)


class HSIFeatureEngineer:
    """
    Feature extraction + RandomForest pipeline, aligned with TB_RFUnet_P.
    - Per-pixel features & labels
    - Streaming to disk to avoid OOM
    """

    def __init__(self, unet_model, num_classes: int, patch_size: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.unet = unet_model.to(device)
        self.unet.eval()
        self.device = device
        self.num_classes = num_classes
        self.patch_size = int(patch_size)

        # RF params exactly as TB_RFUnet_P
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )

    # ----------------------------
    # Feature Extraction (stream)
    # ----------------------------
    def extract_features(self, dataloader: DataLoader, is_train: bool = False,
                         save_dir: str = "temp_features"):
        """
        Stream features/labels to disk per batch.
        Return: (deep_files, depth_files, label_files, patch_indices)
        - label_files may be [] if dataset has no labels (e.g., full-image inference).
        """
        os.makedirs(save_dir, exist_ok=True)

        # Optional: clear old npy to avoid mixing runs
        for f in os.listdir(save_dir):
            if f.endswith(".npy"):
                os.remove(os.path.join(save_dir, f))

        feature_files, depth_files, label_files = [], [], []
        all_indices: List[tuple] = []

        self.unet.eval()
        with torch.no_grad():
            for bidx, batch in enumerate(dataloader):
                # Compatible batch parsing (train/val return 5 items; test 4)
                if len(batch) == 5:
                    hsi_patches, msi_patches, depth_patches, patch_labels, indices = batch
                    has_labels = True
                elif len(batch) == 4:
                    hsi_patches, msi_patches, depth_patches, indices = batch
                    patch_labels = None
                    has_labels = False
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")
                all_indices.extend(indices)

                hsi_patches = hsi_patches.to(self.device).float()
                msi_patches = msi_patches.to(self.device).float()
                depth_patches = depth_patches.to(self.device).float()

                # Forward UNet: features (B, C, H, W), logits unused
                features, _ = self.unet(hsi_patches, msi_patches, depth_patches)

                # Flatten order is CRITICAL: (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
                features = features.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)
                n, h, w, c = features.shape
                feats_2d = features.reshape(n * h * w, c)

                depth_np = depth_patches.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)
                depth_2d = depth_np.reshape(n * h * w, depth_np.shape[-1])

                # Save deep/depth of this batch
                prefix = "train" if is_train else "val_or_test"
                f_path = os.path.join(save_dir, f"{prefix}_deep_{bidx:05d}.npy")
                d_path = os.path.join(save_dir, f"{prefix}_depth_{bidx:05d}.npy")
                np.save(f_path, feats_2d)
                np.save(d_path, depth_2d)
                feature_files.append(f_path)
                depth_files.append(d_path)

                # Labels: ensure per-pixel, no double-repeat
                if has_labels:
                    lab_np = patch_labels.numpy()
                    # If label is per-patch (shape=(B,)), expand; if already (B,H,W) keep
                    if lab_np.ndim == 1 and lab_np.shape[0] == n:
                        lab_px = np.repeat(lab_np.astype(np.int64), h * w)
                    else:
                        # expect (B, H, W) or (H, W) per item collated
                        lab_np = lab_np.reshape(n, h, w)
                        lab_px = lab_np.reshape(-1).astype(np.int64)
                    l_path = os.path.join(save_dir, f"{prefix}_label_{bidx:05d}.npy")
                    np.save(l_path, lab_px)
                    label_files.append(l_path)

                if (bidx + 1) % 10 == 0:
                    print(f"  ↳ Saved {bidx + 1}/{len(dataloader)} batches to disk...")

        return feature_files, depth_files, label_files, all_indices

    # ----------------------------
    # Combine Modalities
    # ----------------------------
    def prepare_combined_features(self,
                                  deep_features: np.ndarray,
                                  raw_features: np.ndarray,
                                  depth_features: np.ndarray,
                                  spectral_relationship_features: np.ndarray) -> np.ndarray:
        # Strict length check
        n = deep_features.shape[0]
        assert raw_features.shape[0] == n, f"raw vs deep mismatch: {raw_features.shape[0]} vs {n}"
        assert depth_features.shape[0] == n, f"depth vs deep mismatch: {depth_features.shape[0]} vs {n}"
        assert spectral_relationship_features.shape[0] == n, \
            f"spectral vs deep mismatch: {spectral_relationship_features.shape[0]} vs {n}"

        return np.hstack([
            deep_features.astype(np.float32),
            raw_features.astype(np.float32),
            depth_features.astype(np.float32),
            spectral_relationship_features.astype(np.float32),
        ])

    # ----------------------------
    # RF Predict
    # ----------------------------
    def predict(self, combined_features: np.ndarray) -> np.ndarray:
        return self.rf_classifier.predict(combined_features)

    def predict_in_batches(self, combined_features: np.ndarray, batch_size: int = 100000) -> np.ndarray:
        preds = []
        N = combined_features.shape[0]
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            preds.append(self.rf_classifier.predict(combined_features[s:e]))
        return np.concatenate(preds, axis=0)

    # ----------------------------
    # Reconstruct Segmentation
    # ----------------------------
    def reconstruct_segmentation(self, predictions: np.ndarray,
                                 patch_indices: List[tuple],
                                 original_shape: tuple) -> np.ndarray:
        H, W = original_shape
        seg = np.zeros((H, W), dtype=np.int64)
        P = self.patch_size
        idx = 0
        for item in patch_indices:
            # item can be (i,j) or (i,j,...) — take first two
            i = item[0]
            j = item[1]
            end_i, end_j = min(i + P, H), min(j + P, W)
            patch_pred = predictions[idx: idx + P * P].reshape(P, P)
            seg[i:end_i, j:end_j] = patch_pred[:end_i - i, :end_j - j]
            idx += P * P
        assert idx <= len(predictions), f"Reconstruct overflow: used {idx}, total {len(predictions)}"
        return seg
