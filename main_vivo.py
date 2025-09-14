#!/usr/bin/env python3
"""
main_vivo.py
------------
DDRM restoration of 512×512 ultrasound RF using a 3-channel
(OpenAI/ImageNet-style) diffusion model.
"""

import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
import scipy.io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from main import parse_args_and_config
from functions.svd_replacement import deconvolution_BCCB
from functions.denoising import efficient_generalized_steps
from runners.diffusion import Diffusion
from guided_diffusion.script_util import create_model


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--degraded_mat", required=True,
                   help=".mat containing 'rf_degraded' (or fallback key)")
    p.add_argument("--gt_mat", default=None,
                   help="(optional) .mat with key 'GT_rf_resized' for metrics")
    p.add_argument("--psf_path", required=True,
                   help=".mat containing PSF (e.g. 'PSF_estim')")
    p.add_argument("--model_path", required=True,
                   help="Pre-trained diffusion checkpoint (.pt)")
    p.add_argument("--config", required=True,
                   help="YAML config (e.g. us512.yml)")
    p.add_argument("--timesteps", type=int, default=20)
    p.add_argument("--eta", type=float, default=0.85)
    p.add_argument("--etaB", type=float, default=1.0)
    p.add_argument("--sigma_0", type=float, default=0.0)
    p.add_argument("--output_dir", default="mat_Restore_DDRM")
    p.add_argument("--doc", default="rf_ddrm")
    return p.parse_args()


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def first_numeric(d):
    for k, v in d.items():
        if not k.startswith("__") and isinstance(v, np.ndarray):
            return v.squeeze(), k
    raise KeyError("No numeric array found")


def load_arr(path, key):
    d = scipy.io.loadmat(path)
    if key in d:
        return d[key].squeeze()
    arr, k = first_numeric(d)
    print(f"⚠️  '{key}' not found in {Path(path).name}; using '{k}'")
    return arr


def pad_center_psf(psf, target_shape):
    H, W = target_shape
    ph, pw = psf.shape
    if (ph, pw) == (H, W):
        return psf.astype(np.float32)
    padded = np.zeros((H, W), dtype=np.float32)
    sh, sw = (H - ph) // 2, (W - pw) // 2
    padded[sh:sh + ph, sw:sw + pw] = psf.astype(np.float32)
    return padded


def build_H_and_singulars(psf_path, img_shape, device, in_ch):
    # Load PSF
    d = scipy.io.loadmat(psf_path)
    if "PSF_estim" in d:
        psf = d["PSF_estim"].squeeze()
    else:
        psf, _ = first_numeric(d)

    psf_pad = pad_center_psf(psf, img_shape)

    # BCCB operator
    H_funcs = deconvolution_BCCB(psf_pad.astype(np.float32), img_shape[0], device)

    # Singular values for DDRM
    psf_fft = torch.fft.fft2(torch.tensor(psf_pad, dtype=torch.complex64, device=device))
    singulars = torch.abs(psf_fft).flatten()
    H_funcs.singulars = lambda: singulars.repeat(in_ch)

    return H_funcs


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load degraded RF
    rf_deg = load_arr(args.degraded_mat, "rf_degraded")
    H, W = rf_deg.shape
    print("RF shape:", (H, W))

    # 2) Parse YAML config (through main.py argument parser)
    sys.argv = [
        "main.py", "--config", args.config, "--doc", args.doc,
        "--ni", "--timesteps", str(args.timesteps),
        "--deg", "deblur_bccb", "--sigma_0", str(args.sigma_0),
        "--eta", str(args.eta), "--etaB", str(args.etaB)
    ]
    run_args, run_cfg = parse_args_and_config()
    betas = Diffusion(run_args, run_cfg, device=device).betas.to(device)

    # 3) Force a 3-channel model to match ImageNet checkpoints
    model_cfg = vars(run_cfg.model)
    model_cfg["in_channels"] = 3
    model_cfg["out_channels"] = 3
    in_ch = 3

    model = create_model(**model_cfg).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model = torch.nn.DataParallel(model).eval()
    print("✅ Diffusion model loaded")

    # 4) Prepare x0 (repeat single-channel RF → pseudo-RGB)
    x0 = torch.tensor(rf_deg, dtype=torch.float32, device=device)[None, None]
    x0 = x0.repeat(1, in_ch, 1, 1)  # (B,3,H,W)

    # 5) Build degradation operator + singular values
    H_funcs = build_H_and_singulars(args.psf_path, (H, W), device, in_ch)
    print("✅ PSF operator and singulars ready")
    
    # 6) DDRM sampling
    step_seq = list(range(0, betas.numel(),
                          max(1, betas.numel() // args.timesteps)))
    x_start = torch.randn_like(x0)
    with torch.no_grad():
        x_seq = efficient_generalized_steps(
            x_start, step_seq, model, betas,
            H_funcs, x0, args.sigma_0 * 2,
            etaA=args.eta, etaB=args.etaB, etaC=args.eta
        )

    # 7) Extract final tensor (unwrap any nested lists)  ──► **NEW**
    x_rest = x_seq
    while isinstance(x_rest, (list, tuple)):            # **NEW**
        x_rest = x_rest[-1]                              # **NEW**
    if torch.is_complex(x_rest):                         # unchanged
        x_rest = x_rest.real                             # unchanged
    rf_rest = x_rest[0, 0].cpu().numpy()                # unchanged

    out_path = Path(args.output_dir) / "restored_ddrm.mat"
    scipy.io.savemat(out_path, {"rf_restored": rf_rest})
    print(f"✅ Saved {out_path}")

    # 8) Optional metrics (unchanged)
    if args.gt_mat:
        rf_gt = load_arr(args.gt_mat, "GT_rf_resized").astype(np.float32)
        psnr = peak_signal_noise_ratio(rf_gt, rf_rest, data_range=2.0)
        ssim = structural_similarity(rf_gt, rf_rest, data_range=2.0)
        print(f"PSNR: {psnr:.2f} dB   SSIM: {ssim:.4f}")



   

if __name__ == "__main__":
    main()
