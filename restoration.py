#!/usr/bin/env python3
"""
run_ddrm_restoration.py
-----------------------
Restores a degraded RF image with DDRM and saves the result.

Inputs
------
--degraded_mat   path/to/degraded_y0.mat   (must contain 'rf_degraded')
--psf_path       path/to/psf_*.mat
--model_path     pretrained diffusion checkpoint
--config         YAML config used to train the model

Optional
--------
--gt_mat         path/to/ground_truth.mat  (for PSNR/SSIM)
--timesteps      DDRM inference steps       [default: 20]
--eta, --etaB    Noise scheduling params    [defaults: 0.85, 1.0]
--output_dir     folder for restored .mat   [default: mat_Restore_DDRM]

Output
------
<output_dir>/restored_ddrm.mat   (variable name: rf_restored)
If --gt_mat is supplied, PSNR and SSIM are printed to stdout.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import scipy.io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from main import parse_args_and_config
from functions.svd_replacement import deconvolution_BCCB
from functions.denoising import efficient_generalized_steps
from runners.diffusion import Diffusion
from guided_diffusion.script_util import create_model, create_model_2


def get_args():
    p = argparse.ArgumentParser(description="DDRM restore degraded RF")
    p.add_argument("--degraded_mat", required=True, type=str,
                   help=".mat produced by degradation (rf_degraded)")
    p.add_argument("--gt_mat",       default=None, type=str,
                   help="(optional) ground-truth .mat for metrics")
    p.add_argument("--psf_path",     required=True, type=str,
                   help="Path to PSF .mat")
    p.add_argument("--model_path",   required=True, type=str,
                   help="Pretrained diffusion checkpoint")
    p.add_argument("--config",       required=True, type=str,
                   help="YAML config used to train model")
    p.add_argument("--timesteps",    default=20, type=int,
                   help="DDRM inference steps")
    p.add_argument("--eta",          default=0.85, type=float,
                   help="Noise scheduling eta")
    p.add_argument("--etaB",         default=1.0, type=float,
                   help="Noise scheduling etaB")
    p.add_argument("--sigma_0",      default=0.0, type=float,
                   help="Noise std-dev used in degradation step")
    p.add_argument("--output_dir",   default="mat_Restore_DDRM", type=str)
    p.add_argument("--doc",          default="rf_ddrm", type=str)
    return p.parse_args()


def first_numeric(mat_dict):
    for k, v in mat_dict.items():
        if not k.startswith("__") and isinstance(v, np.ndarray):
            return v.squeeze(), k
    raise KeyError("No numeric array in MAT file")


def load_arr(mat_path, prefer_key):
    d = scipy.io.loadmat(mat_path)
    if prefer_key in d:
        return d[prefer_key].squeeze()
    arr, k = first_numeric(d)
    print(f"⚠️  '{prefer_key}' not found in {mat_path}, using key '{k}'")
    return arr


def build_H(psf_path, im_size, device):
    psf_mat = scipy.io.loadmat(psf_path)
    psf, k = first_numeric(psf_mat)
    print(f"✅ PSF loaded (key='{k}', shape={psf.shape})")
    return deconvolution_BCCB(psf.astype(np.float32), im_size, device)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load degraded RF
    rf_deg = load_arr(args.degraded_mat, "rf_degraded")  # (H, W)
    H, W = rf_deg.shape
    img_sz = max(H, W)
    rf_t = torch.tensor(rf_deg).unsqueeze(0).unsqueeze(0)
    rf_t = torch.nn.functional.interpolate(
        rf_t, size=(img_sz, img_sz), mode="bilinear", align_corners=False
    )
    x0 = rf_t.repeat(1, 3, 1, 1).to(device) * 2 - 1  # (1,3,img_sz,img_sz)

    # 2) Build degradation operator
    H_funcs = build_H(args.psf_path, img_sz, device)

    # 3) Parse config
    sys.argv = [
        "main.py", "--config", args.config, "--doc", args.doc,
        "--ni", "--timesteps", str(args.timesteps),
        "--deg", "deblur_bccb", "--sigma_0", str(args.sigma_0),
        "--eta", str(args.eta), "--etaB", str(args.etaB)
    ]
    run_args, run_cfg = parse_args_and_config()
    betas = Diffusion(run_args, run_cfg, device=device).betas.to(device)

    # 4) Load pretrained model
    cfg = vars(run_cfg.model)
    if run_cfg.model.type.lower() == "openai":
        model = create_model(**cfg)
    else:
        model = create_model_2(**cfg).model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = torch.nn.DataParallel(model).to(device).eval()
    print("✅ Model checkpoint loaded")

    # 5) DDRM sampling
    step_seq = list(range(0, betas.numel(), max(1, betas.numel() // args.timesteps)))
    x_start = torch.randn_like(x0)
    with torch.no_grad():
        x_seq = efficient_generalized_steps(
            x_start, step_seq, model, betas,
            H_funcs, x0, args.sigma_0 * 2,
            etaA=args.eta, etaB=args.etaB, etaC=args.eta
        )

    # Unwrap and real part
    x_rest = x_seq
    while isinstance(x_rest, (list, tuple)):
        x_rest = x_rest[-1]
    if torch.is_complex(x_rest):
        x_rest = x_rest.real

    # 6) Convert to numpy RF in [0,1]
    rf_rest = ((x_rest[:, :1] + 1) * 0.5)[0, 0].cpu().numpy()

    # Save without normalization
    out_file = Path(args.output_dir) / "restored_ddrm.mat"
    scipy.io.savemat(out_file, {"rf_restored": rf_rest})
    print(f"✅ Saved {out_file} (unnormalized)")

    # 7) Optional PSNR & SSIM
    if args.gt_mat:
        rf_gt = load_arr(args.gt_mat, "GT_rf_resized")
        rf_gt = torch.tensor(rf_gt).unsqueeze(0).unsqueeze(0)
        rf_gt = torch.nn.functional.interpolate(
            rf_gt, size=(img_sz, img_sz), mode="bilinear", align_corners=False
        )[0, 0].numpy()

        psnr = peak_signal_noise_ratio(rf_gt, rf_rest, data_range=rf_gt.max() - rf_gt.min())
        ssim = structural_similarity(rf_gt, rf_rest, data_range=rf_gt.max() - rf_gt.min())
        print(f"PSNR: {psnr:0.2f} dB   SSIM: {ssim:0.4f}")


if __name__ == "__main__":
    main()
