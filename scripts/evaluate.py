"""Compute accuracy on a saved model."""
import argparse, tensorflow as tf, yaml
from src.data.dataset_loader import load_dataset

def evaluate(model_path: str, cfg_path: str):
    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    ds_val = load_dataset(cfg['paths']['dataset_root'],
                          cfg['model']['img_size'],
                          cfg['model']['batch_size'])[1]
    model = tf.keras.models.load_model(model_path)
    loss, acc = model.evaluate(ds_val, verbose=0)
    print(f"Validation accuracy: {acc:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args()
    evaluate(args.model, args.config)
