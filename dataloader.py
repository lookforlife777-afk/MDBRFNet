import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize
import rasterio

class HyperspectralDataset(Dataset):
    def __init__(self, hsi_image, msi_image, depth_feature, labels=None, patch_size=32, stride=24):
        self.hsi_image = hsi_image
        self.msi_image = msi_image
        self.depth_feature = depth_feature
        self.labels = labels
        self.patch_size = patch_size
        self.stride = stride

        self.patches, self.msi_patches, self.depth_patches, self.patch_labels, self.patch_indices = self._extract_patches()

    def _extract_patches(self):
        """完全复刻 TB_RFUnet_P.py 的 patch 提取逻辑"""
        height, width, channels = self.hsi_image.shape
        patches, msi_patches, depth_patches, patch_labels, patch_indices = [], [], [], [], []

        for i in range(0, height, self.stride):
            for j in range(0, width, self.stride):
                end_i = min(i + self.patch_size, height)
                end_j = min(j + self.patch_size, width)

                # --- 提取原始 patch ---
                patch = self.hsi_image[i:end_i, j:end_j, :]
                msi_patch = self.msi_image[i:end_i, j:end_j, :]
                depth_patch = self.depth_feature[i:end_i, j:end_j, :]

                # --- 创建填充后的固定大小 patch ---
                padded_patch = np.zeros((self.patch_size, self.patch_size, channels), dtype=self.hsi_image.dtype)
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                patches.append(padded_patch)

                padded_msi_patch = np.zeros((self.patch_size, self.patch_size, self.msi_image.shape[-1]),
                                            dtype=self.msi_image.dtype)
                padded_msi_patch[:msi_patch.shape[0], :msi_patch.shape[1], :] = msi_patch
                msi_patches.append(padded_msi_patch)

                padded_depth_patch = np.zeros((self.patch_size, self.patch_size, self.depth_feature.shape[-1]),
                                              dtype=self.depth_feature.dtype)
                padded_depth_patch[:depth_patch.shape[0], :depth_patch.shape[1], :] = depth_patch
                depth_patches.append(padded_depth_patch)

                # --- 标签补齐 ---
                if self.labels is not None:
                    label_patch = self.labels[i:end_i, j:end_j]
                    padded_label_patch = np.zeros((self.patch_size, self.patch_size), dtype=self.labels.dtype)
                    padded_label_patch[:label_patch.shape[0], :label_patch.shape[1]] = label_patch
                    patch_labels.append(padded_label_patch)

                patch_indices.append((i, j))

        patches = np.array(patches)
        msi_patches = np.array(msi_patches)
        depth_patches = np.array(depth_patches)

        if self.labels is not None:
            patch_labels = np.array(patch_labels)
            return patches, msi_patches, depth_patches, patch_labels, patch_indices
        else:
            return patches, msi_patches, depth_patches, None, patch_indices

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        hsi_patch = torch.tensor(self.patches[idx]).permute(2, 0, 1)
        msi_patch = torch.tensor(self.msi_patches[idx]).permute(2, 0, 1)
        depth_patch = torch.tensor(self.depth_patches[idx]).permute(2, 0, 1)
        index = self.patch_indices[idx]

        if self.labels is not None:
            patch_label = torch.tensor(self.patch_labels[idx], dtype=torch.long)
            return hsi_patch, msi_patch, depth_patch, patch_label, index
        else:
            return hsi_patch, msi_patch, depth_patch, index


def load_data(img_path, dep_data_path, msi_data_path, label_path):
    """完全同步 TB_RFUnet_P.py 的数据加载流程"""
    # === 1. 读取影像数据 ===
    with rasterio.open(img_path) as src:
        data1 = src.read().transpose(1, 2, 0)
        data1 = np.nan_to_num(data1, nan=0)
        profile = src.profile

    with rasterio.open(msi_data_path) as src:
        data2 = src.read().transpose(1, 2, 0)
        data2 = np.nan_to_num(data2, nan=0)

    with rasterio.open(dep_data_path) as src:
        data3 = src.read().transpose(1, 2, 0)
        data3 = np.nan_to_num(data3, nan=0)

    with rasterio.open(label_path) as src:
        labels = src.read(1)
        labels = np.nan_to_num(labels, nan=0)

    # === 2. 边界填充 ===
    padding = 16
    hsi_image = np.pad(data1, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    msi_image = np.pad(data2, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    depth_feature = np.pad(data3, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    labels = np.pad(labels, ((padding, padding), (padding, padding)), mode='constant')

    hsi_height, hsi_width = hsi_image.shape[:2]

    # === 3. 多光谱重采样 ===
    print('resize msi data from', data2.shape, 'to', (hsi_height, hsi_width, data2.shape[-1]))
    resampled_msi = resize(
        msi_image, (hsi_height, hsi_width),
        order=1, anti_aliasing=True
    )

    # === 4. 水深重采样 ===
    if data3.shape[:2] != (hsi_height, hsi_width):
        print('resize depth data from', data3.shape, 'to', (hsi_height, hsi_width))
        resampled_depth = resize(
            depth_feature, (hsi_height, hsi_width),
            order=1, anti_aliasing=True
        )
    else:
        resampled_depth = depth_feature

    # === 5. 标签重采样（最近邻） ===
    if labels.shape != (hsi_height, hsi_width):
        print('resize label data from', labels.shape, 'to', (hsi_height, hsi_width))
        labels = resize(
            labels, (hsi_height, hsi_width),
            order=0, anti_aliasing=False, preserve_range=True
        ).astype(np.int64)

    # === 6. 返回与原始完全一致的数据结构 ===
    return hsi_image[:, :, :100], resampled_msi, resampled_depth, labels, profile
