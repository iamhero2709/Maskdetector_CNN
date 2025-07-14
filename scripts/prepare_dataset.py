#!/usr/bin/env python
"""
Convert Pascal‑VOC style images+annotations into our
  data/face_mask/{with_mask,without_mask}/  folders.

Example:
$ python scripts/prepare_dataset.py \
      --src ../my-dataset \
      --dst data/face_mask
"""
import argparse, shutil
from pathlib import Path
import xml.etree.ElementTree as ET

CLASS_MAP = {
    "mask":        "with_mask",
    "with_mask":   "with_mask",
    "no_mask":     "without_mask",
    "without_mask":"without_mask",
    "mask_weared_incorrect": "without_mask"   # you can keep it separate if you like
}

def main(src_root: Path, dst_root: Path, dry_run=False):
    img_dir = src_root / "images"
    ann_dir = src_root / "annotations"

    assert img_dir.exists() and ann_dir.exists(), "images/ or annotations/ missing!"

    for xml_file in ann_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # Grab the *first* <object><name> tag
        label = root.findtext("./object/name")
        if label is None:
            print(f"[skip] No <name> tag in {xml_file.name}")
            continue

        label = CLASS_MAP.get(label.lower())
        if label is None:
            print(f"[skip] Unknown class '{label}' in {xml_file.name}")
            continue

        dst_class_dir = dst_root / label
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        img_name = root.findtext("filename")
        img_path = img_dir / img_name
        if not img_path.exists():
            print(f"[warn] Image {img_name} missing for {xml_file.name}")
            continue

        dst_path = dst_class_dir / img_name
        if dry_run:
            print(f"Would copy {img_path} -> {dst_path}")
        else:
            shutil.copy2(img_path, dst_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset root with images/ and annotations/")
    ap.add_argument("--dst", default="data/face_mask", help="output root")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    main(Path(args.src), Path(args.dst), dry_run=args.dry_run)
