# MDBRFNet_full

This repository contains a full, runnable project skeleton.

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
- Designed for Windows.
