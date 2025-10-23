import numpy as np

def calculate_spectral_relationship_features(patches: np.ndarray) -> np.ndarray:

    # 波段均值 (indexes follow original script assumptions)
    blue = np.mean(patches[:, :, :, 7:14], axis=3)
    green = np.mean(patches[:, :, :, 16:23], axis=3)
    red = np.mean(patches[:, :, :, 28:33], axis=3)
    nir = np.mean(patches[:, :, :, 52:58], axis=3)
    swir = np.mean(patches[:, :, :, 70:76], axis=3)

    epsilon = 1e-2
    ndvi = np.nan_to_num((nir - red) / (nir + np.abs(red) + epsilon))
    rvi = np.nan_to_num(nir / (np.abs(red) + epsilon))
    dvi = np.nan_to_num(nir - red)
    ndwi = np.nan_to_num((green - nir) / (np.abs(green) + np.abs(nir) + epsilon))
    twi = np.nan_to_num(red - swir)
    cmi = np.nan_to_num(green - blue - (swir - blue) * 0.2)
    fai = np.nan_to_num(nir - red - (swir - red) * 0.5)

    spectral_relationship_features = np.stack([ndvi, rvi, dvi, ndwi, twi, cmi, fai], axis=-1)
    num_patches, width, height, channels = spectral_relationship_features.shape
    spectral_relationship_features = spectral_relationship_features.reshape(-1, channels)
    return spectral_relationship_features
