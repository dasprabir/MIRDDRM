#!/usr/bin/env python3
"""
apply_degradation.py
--------------------
Create a degraded RF image from the ground-truth RF contained in a .mat file.
Saves:
    mat_output_rf/ground_truth.mat   (variable: GT_rf_resized)
    mat_output_rf/degraded_y0.mat    (variable: rf_degraded)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import scipy.io

from datasets import get_ultrasound_mat_dataset
from functions.svd_replacement import deconvolution_BCCB


# ------------------------------------------------------------------ #
# Argument parsing
# ------------------------------------------------------------------ #
def get_args():
    parser = argparse.ArgumentParser(description="Apply PSF degradation to RF data")
    parser.add_argument("--mat_dir",   required=True, type=str, help="Folder with data.mat")
    parser.add_argument("--mat_list",  required=True, type=str, help="Text file listing .mat names")
    parser.add_argument("--key",       default="GT_rf", type=str, help="Variable name in .mat")
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--psf_path",  required=True, type=str, help="Full path to psf_*.mat")
    parser.add_argument("--sigma_0",   default=0.0, type=float, help="Std-dev of added Gaussian noise")
    parser.add_argument("--output_dir", default="mat_output_rf", type=str)
    return parser.parse_args()


# ------------------------------------------------------------------ #
# Helper: robust PSF extraction
# ------------------------------------------------------------------ #
def extract_psf(mat_dict, preferred_keys=("psf_ref", "PSF_estim", "psf", "psf_GT")):
    """Return first matching key or first numeric array if none match."""
    for k in preferred_keys:
        if k in mat_dict:
            print(f"✅ Found PSF key: {k}")
            return mat_dict[k]

    # otherwise pick the first non-meta ndarray
    for k, v in mat_dict.items():
        if not k.startswith("__") and isinstance(v, np.ndarray):
            print(f"⚠️  Fallback: using first numeric array in MAT file (key='{k}')")
            return v

    raise KeyError("No numeric array found in PSF .mat file.")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load ground-truth RF
    ds = get_ultrasound_mat_dataset(
        args.mat_dir, args.mat_list, key=args.key, image_size=args.image_size
    )
    x_gray, _ = ds[0]                          # (1, H, W) torch tensor in [0,1]
    x = x_gray.unsqueeze(0).repeat(1, 3, 1, 1) # (1,3,H,W)
    x = x.to(device) * 2 - 1                   # scale to [-1,1]

    # 2) Load PSF and build BCCB operator
    psf_mat = scipy.io.loadmat(args.psf_path)
    psf     = extract_psf(psf_mat).astype(np.float32)
    H_funcs = deconvolution_BCCB(psf, args.image_size, device)

    # 3) Apply degradation H(x) + noise
    y0 = H_funcs.H(x)
    if torch.is_complex(y0):
        y0 = y0.real
    if args.sigma_0 > 0:
        y0 = y0 + args.sigma_0 * torch.randn_like(y0)

    # 4) Save outputs
    gt_np       = x[0, 0].cpu().numpy()
    degraded_np = y0.view_as(x)[0, 0].cpu().numpy()

    scipy.io.savemat(Path(args.output_dir) / "ground_truth.mat",
                     {"GT_rf_resized": gt_np})
    scipy.io.savemat(Path(args.output_dir) / "degraded_y0.mat",
                     {"rf_degraded": degraded_np})

    print("✅ Saved:")
    print("   • ground_truth.mat   (GT_rf_resized)")
    print("   • degraded_y0.mat    (rf_degraded)")


if __name__ == "__main__":
    main()
