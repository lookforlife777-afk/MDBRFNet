import os
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from data.dataloader import HyperspectralDataset, load_data
from models.dual_branch_unet import DualBranchUNet
from utils.feature_engineer import HSIFeatureEngineer
from utils.spectral_features import calculate_spectral_relationship_features
from sklearn.impute import SimpleImputer
from sklearn.utils import check_array
import matplotlib.pyplot as plt
torch.manual_seed(42)
np.random.seed(42)

def run_inference(config: dict):
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
    feature_dim = model_cfg.get('feature_dim', 64)

    os.makedirs(output_dir, exist_ok=True)

    hsi_image, msi_image, depth_feature, labels, profile = load_data(hsi_path, depth_path, msi_path, label_path)
    assert hsi_image.shape[:2] == msi_image.shape[:2] == depth_feature.shape[:2], "数据尺寸不匹配"

    # create train set and dataloader (half patches used as training in original script)
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

    # initialize unet and optionally load pretrained weights if provided
    num_classes = len(np.unique(labels))
    unet = DualBranchUNet(hsi_channels=hsi_image.shape[-1], msi_channels=msi_image.shape[-1], num_classes=num_classes, feature_dim=feature_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet.to(device)

    # pretrain step in original script: we include an option to load saved model if exists
    model_path = os.path.join(output_dir, 'tbunet_hsi_model.pth')
    if os.path.exists(model_path):
        print(f"加载已有 U-Net 模型: {model_path}")
        unet.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("未找到预训练模型，请先运行 train 模式或在 outputs 中放入模型。将会使用未经训练的网络进行特征提取。")


    # initialize feature engineer
    engineer = HSIFeatureEngineer(unet, num_classes=num_classes, patch_size=patch_size)

    print("提取深度学习特征 (训练集)...")
    deep_features, depth_features, patch_labels, patch_indices = engineer.extract_features(train_dataloader, is_train=True)

    print("提取原始特征...")
    raw_features = train_dataset.patches.reshape(-1, train_dataset.patches.shape[-1])

    print("计算谱间关系特征...")
    spectral_relationship_features = calculate_spectral_relationship_features(train_dataset.patches)

    print("组合特征...")
    combined_features = engineer.prepare_combined_features(deep_features, raw_features, depth_features, spectral_relationship_features)

    combined_features = check_array(combined_features, ensure_all_finite='allow-nan')
    combined_features[np.isinf(combined_features)] = np.nan
    imputer = SimpleImputer(strategy='mean')
    combined_features = imputer.fit_transform(combined_features)

    assert combined_features.shape[0] == len(train_indices) * patch_size * patch_size, "融合特征样本数量不正确"

    print("训练随机森林分类器...")
    engineer.train_random_forest(combined_features, patch_labels)

    # feature importance plot (as in original script)
    importances = engineer.rf_classifier.feature_importances_
    indices = np.argsort(importances)
    top_indices = indices[-50:][::-1]

    # compute feature sources mapping
    feature_dim_count = feature_dim
    orig_count = hsi_image.shape[-1]
    spectral_count = spectral_relationship_features.shape[-1]
    feature_sources = []
    for i in range(importances.shape[0]):
        if i < feature_dim_count:
            feature_sources.append('Advanced features')
        elif i < feature_dim_count + orig_count:
            feature_sources.append('Original features')
        elif i < feature_dim_count + orig_count + 1:
            feature_sources.append('Depth features')
        else:
            feature_sources.append('Spectral relationship features')

    selected_importances = importances[top_indices]
    selected_sources = [feature_sources[i] for i in top_indices]
    color_map = {'Advanced features':'red','Original features':'green','Depth features':'blue','Spectral relationship features':'yellow'}
    colors = [color_map[s] for s in selected_sources]

    feature_labels = []
    for i, idx in enumerate(top_indices):
        if idx < feature_dim_count:
            feature_type = 'Advanced features'
            feature_idx = idx + 1
        elif idx < feature_dim_count + orig_count:
            feature_type = 'Original features'
            feature_idx = idx - feature_dim_count + 1
        elif idx < feature_dim_count + orig_count + 1:
            feature_type = 'Depth features'
            feature_idx = idx - feature_dim_count - orig_count + 1
        else:
            feature_type = 'Spectral relationship features'
            feature_idx = idx - feature_dim_count - orig_count - 1 + 1
        feature_labels.append(f"{feature_type}-{feature_idx}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    bars = plt.bar(range(len(top_indices)), selected_importances, color=colors, align='center', alpha=0.7)
    plt.title('Top 30 Feature Importances')
    plt.ylabel('Importance')
    plt.xticks(range(len(top_indices)), feature_labels, rotation=90)
    legend_elements = [
        plt.Rectangle((0,0),1,1,color=color_map['Advanced features'],alpha=0.7,label='Advanced features'),
        plt.Rectangle((0,0),1,1,color=color_map['Original features'],alpha=0.7,label='Original features'),
        plt.Rectangle((0,0),1,1,color=color_map['Depth features'],alpha=0.7,label='Depth features'),
        plt.Rectangle((0,0),1,1,color=color_map['Spectral relationship features'],alpha=0.7,label='Spectral relationship features'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.show()

    # Generate full-image predictions
    print("提取全图深度学习特征...")
    full_image_dataset = HyperspectralDataset(hsi_image, msi_image, depth_feature, labels=None, patch_size=patch_size, stride=patch_size)
    full_image_dataloader = DataLoader(full_image_dataset, batch_size=batch_size, shuffle=False)
    full_deep_features, full_depth_features, full_patch_indices = engineer.extract_features(full_image_dataloader, is_train=False)
    full_raw_features = full_image_dataset.patches.reshape(-1, full_image_dataset.patches.shape[-1])
    full_spectral_relationship_features = calculate_spectral_relationship_features(full_image_dataset.patches)
    full_combined_features = engineer.prepare_combined_features(full_deep_features, full_raw_features, full_depth_features, full_spectral_relationship_features)
    full_combined_features = check_array(full_combined_features, ensure_all_finite='allow-nan')
    full_combined_features[np.isinf(full_combined_features)] = np.nan
    full_combined_features = SimpleImputer(strategy='mean').fit_transform(full_combined_features)

    print("预测和重构分割结果...")
    full_predictions = engineer.predict(full_combined_features)
    full_segmentation = engineer.reconstruct_segmentation(full_predictions, full_patch_indices, hsi_image.shape[:2])

    mask = labels != 0
    print('full image acc:', np.sum(full_segmentation == labels) / np.size(labels))
    print('full image mask acc:', np.sum(full_segmentation[mask] == labels[mask]) / np.size(labels[mask]))

    # 保存结果为 tif
    print("保存分割结果为 tif 文件...")
    profile.update(dtype=rasterio.int64, count=1)
    out_path = os.path.join(output_dir, 'hsi_full_segmentation_tbrfunet.tif')
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(full_segmentation.astype(rasterio.int64), 1)
    print(f"保存为: {out_path}")
