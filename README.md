# MDBRFNet_full

This project is designed for remote sensing multi-modal pixel-level classification, based on the improved Unet and RF classifiers.

## Usage (command-line mode)
- Train U-Net and RF pipeline:
  ```bash
  python main.py --mode train --config configs/default.yaml
  ```
- Run inference (requires trained model and/or RF):
  ```bash
  python main.py --mode inference --config configs/default.yaml
  ```

## Notes
- The `data/` folder doesn't include datasets. Put your TIFF data as configured in `configs/default.yaml`.
- More codes and data will be made public in the future.
- Designed for Windows.
