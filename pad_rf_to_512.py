#!/usr/bin/env python3
"""
pad_rf_to_512.py  –  ZERO-pad RF frames to 512×512
--------------------------------------------------
Recursively scans a folder, loads RF frames (.mat, .npy, .png, …),
centres them on a 512×512 canvas, and saves the result as .npy or .mat.

Example
-------
python pad_rf_to_512.py \
    --src_dir  raw_rf_frames \
    --dst_dir  rf_train_512 \
    --in_key   rf \
    --out_type npy
"""

import argparse, sys
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image
from tqdm import tqdm


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", required=True, type=str,
                   help="Folder with RF files (searches recursively by default)")
    p.add_argument("--dst_dir", required=True, type=str,
                   help="Output folder for 512×512 files")
    p.add_argument("--in_key",  default="rf", type=str,
                   help="Variable name inside .mat (default: rf)")
    p.add_argument("--out_type", default="npy", choices=["npy", "mat"],
                   help="Write .npy (fast) or .mat (MATLAB)")
    p.add_argument("--recursive", action="store_true", default=True,
                   help="Recurse into sub-directories (default: on)")
    return p.parse_args()


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
EXT = {".mat", ".MAT", ".npy", ".npz",
       ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_rf(path: Path, in_key: str):
    suf = path.suffix
    if suf.lower() in {".npy"}:
        return np.load(path)
    if suf.lower() in {".npz"}:
        return np.load(path)["arr_0"]
    if suf.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return np.array(Image.open(path)).astype(np.float32)
    if suf.lower() == ".mat":
        d = scipy.io.loadmat(path)
        if in_key not in d:
            raise KeyError(f"'{in_key}' not found in {path.name}")
        return d[in_key].squeeze()
    raise ValueError(f"Unsupported file type: {path.name}")


def save_rf(arr: np.ndarray, path: Path, out_type: str):
    if out_type == "npy":
        np.save(path.with_suffix(".npy"), arr.astype(np.float32))
    else:
        scipy.io.savemat(path.with_suffix(".mat"), {"rf": arr.astype(np.float32)})


def pad_to_512(arr: np.ndarray) -> np.ndarray:
    H, W = arr.shape
    if H > 512 or W > 512:
        raise ValueError(f"Frame {arr.shape} exceeds 512 pixels")
    pad_h = (512 - H) // 2
    pad_w = (512 - W) // 2
    out = np.zeros((512, 512), dtype=arr.dtype)
    out[pad_h:pad_h + H, pad_w:pad_w + W] = arr
    return out


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    args   = parse_args()
    src    = Path(args.src_dir)
    dst    = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if args.recursive else "*"
    files = [f for f in src.glob(pattern) if f.suffix in EXT]

    if not files:
        print("No supported RF files found. "
              "Check --src_dir or file extensions.")
        sys.exit(0)

    n_ok, n_skip = 0, 0
    for f in tqdm(files, desc="Padding RF"):
        try:
            rf      = load_rf(f, args.in_key)
            rf_pad  = pad_to_512(rf)
            save_rf(rf_pad, dst / f.stem, args.out_type)
            n_ok += 1
        except Exception as e:
            print(f"⚠️  {f.name} skipped: {e}")
            n_skip += 1

    print(f"✅ Done.  Padded: {n_ok}   Skipped: {n_skip}   → {dst}  ({args.out_type})")


if __name__ == "__main__":
    main()
