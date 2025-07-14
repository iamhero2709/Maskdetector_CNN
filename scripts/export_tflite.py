"""Export the best model to TFLite."""
import argparse, tensorflow as tf

def export(model_path: str, out_path: str = "maskdet.tflite"):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(out_path, "wb") as f: f.write(tflite_model)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="maskdet.tflite")
    args = ap.parse_args()
    export(args.model, args.out)
