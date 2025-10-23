import yaml
import argparse
from train import train_model
from inference import run_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="config file path")
    parser.add_argument("--mode", choices=["train", "inference"], default="train", help="mode: train or inference")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.mode == "train":
        train_model(config)
    else:
        run_inference(config)
