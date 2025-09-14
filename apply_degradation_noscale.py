#!/usr/bin/env python3
"""
apply_degradation_noscale.py
----------------------------
Apply PSF degradation to raw RF data without resizing.
Saves:
    mat_output_rf/ground_truth.mat   (variable: GT_rf)
    mat_output_rf/degraded_y0.mat    (variable: rf_degraded)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import scipy.io
import torch
from functions.svd_replacement import deconvolution_BCCB


def get_args():
    parser = argparse.ArgumentParser(description="Apply PSF degradation to RF data (no resize)")
    parser.add_argument("--mat_path", required=True, type=str, help="Path to .mat file containing GT_rf")
    parser.add_argument("--key", default="GT_rf", type=str, help="Variable name in .mat")
    parser.add_argument("--psf_path", required=True, type=str, help="Path to .mat file with PSF")
    parser.add_argument("--sigma_0", default=0.0, type=float, help="Gaussian noise std-dev")
    parser.add_argument("--output_dir", default="mat_output_rf", type=str)
    return parser.parse_args()


def extract_psf(mat_dict, preferred_keys=("psf_ref", "PSF_estim", "psf", "psf_GT")):
    for k in preferred_keys:
        if k in mat_dict:
            print(f"✅ Found PSF key: {k}")
            return mat_dict[k]

    for k, v in mat_dict.items():
        if not k.startswith("__") and isinstance(v, np.ndarray):
            print(f"⚠️  Fallback: using first numeric array in MAT file (key='{k}')")
            return v

    raise KeyError("No numeric array found in PSF .mat file.")


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load RF
    mat_data = scipy.io.loadmat(args.mat_path)
    rf = mat_data.get(args.key)
    if rf is None:
        raise KeyError(f"❌ Key '{args.key}' not found in {args.mat_path}")
    print(f"✅ Loaded RF shape: {rf.shape}")

    # Normalize RF to [-1, 1]
    rf = rf.astype(np.float32)
    rf_norm = (rf - rf.min()) / (rf.max() - rf.min() + 1e-8)
    x = torch.tensor(rf_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device) * 2 - 1

    # Load PSF
    psf_data = scipy.io.loadmat(args.psf_path)
    psf = extract_psf(psf_data).astype(np.float32)

    # Apply BCCB degradation
    H_funcs = deconvolution_BCCB(psf, rf.shape[0], device)
    y0 = H_funcs.H(x)
    if torch.is_complex(y0):
        y0 = y0.real
    if args.sigma_0 > 0:
        y0 += args.sigma_0 * torch.randn_like(y0)

    # Save results
    scipy.io.savemat(Path(args.output_dir) / "ground_truth.mat", {"GT_rf": rf})
    scipy.io.savemat(Path(args.output_dir) / "degraded_y0.mat", {"rf_degraded": y0[0, 0].cpu().numpy()})

    print("✅ Degradation complete. Files saved to:", args.output_dir)


if __name__ == "__main__":
    main()
