#!/usr/bin/env python3
"""
run_ddrm_vivo.py (modified)
----------------------------
Apply DDRM restoration directly on in-vivo RF data, using precomputed
256×256 RF and PSF inputs:

  --rf_mat     path/to/rf_extended.mat   (must contain 'rf_extended')
  --psf_mat    path/to/psf_norm.mat      (must contain 'psf_norm')
  --model_path pretrained diffusion checkpoint
  --config     YAML config used to train the model

Options:
  --timesteps  DDRM inference steps      [default: 20]
  --eta        noise schedule param      [default: 0.85]
  --etaB       secondary noise param     [default: 1.0]
  --sigma_0    noise std-dev in degr.    [default: 0.0]
  --output_dir folder for restored .mat  [default: mat_Restore_VIVO]

Output:
  <output_dir>/restored_ddrm.mat   with variable name 'rf_restored'
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import scipy.io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from main      import parse_args_and_config
from functions.svd_replacement import deconvolution_BCCB
from functions.denoising        import efficient_generalized_steps
from runners.diffusion          import Diffusion
from guided_diffusion.script_util import create_model, create_model_2

# ---------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser(description="DDRM on vivo RF (256×256 inputs)")
    p.add_argument("--rf_mat",     required=True, type=str,
                   help=".mat file containing 'rf_extended' (256x256)")
    p.add_argument("--psf_mat",    required=True, type=str,
                   help=".mat file containing 'psf_norm' (PSF normalized)")
    p.add_argument("--model_path", required=True, type=str,
                   help="Path to pretrained diffusion model checkpoint")
    p.add_argument("--config",     required=True, type=str,
                   help="YAML config used to train the model")
    p.add_argument("--timesteps",  default=20,   type=int,
                   help="Number of DDRM inference steps")
    p.add_argument("--eta",        default=0.85, type=float,
                   help="Noise schedule parameter eta")
    p.add_argument("--etaB",       default=1.0,  type=float,
                   help="Secondary noise schedule parameter etaB")
    p.add_argument("--sigma_0",    default=0.0,  type=float,
                   help="Std-dev of noise in degradation model")
    p.add_argument("--output_dir", default="mat_Restore_VIVO", type=str,
                   help="Directory for restored output")
    return p.parse_args()

# ---------------------------------------------------------------------------- #
def first_numeric(mat_dict):
    for k, v in mat_dict.items():
        if not k.startswith("__") and isinstance(v, np.ndarray):
            return v.squeeze(), k
    raise KeyError("No numeric array found in MAT file")


def load_arr(mat_path, prefer_key):
    d = scipy.io.loadmat(mat_path)
    if prefer_key in d:
        return d[prefer_key].squeeze()
    arr, k = first_numeric(d)
    print(f"⚠️  '{prefer_key}' not found in {mat_path}, using '{k}'")
    return arr


def build_H(psf_path, im_size, device):
    psf_mat = scipy.io.loadmat(psf_path)
    psf, k  = first_numeric(psf_mat)
    print(f"✅ PSF loaded (key='{k}', shape={psf.shape})")
    return deconvolution_BCCB(psf.astype(np.float32), im_size, device)

# ---------------------------------------------------------------------------- #
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # 1) Load 256×256 RF and PSF
    # ------------------------------------------------------------------ #
    rf_ext = load_arr(args.rf_mat, "rf_extended")   # (256,256)
    H, W  = rf_ext.shape
    img_sz = H  # =256

    # convert to tensor and expand to 3 channels
    rf_t = torch.tensor(rf_ext, dtype=torch.float32)
    rf_t = rf_t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device).clamp(-1, 1)

    # ------------------------------------------------------------------ #
    # 2) Build degradation operator from PSF
    # ------------------------------------------------------------------ #
    H_funcs = build_H(args.psf_mat, img_sz, device)

    # ------------------------------------------------------------------ #
    # 3) Parse config & prepare betas
    # ------------------------------------------------------------------ #
    sys.argv = ["main.py", "--config", args.config, "--doc", "vivo_ddrm",
                "--ni", "--timesteps", str(args.timesteps),
                "--deg", "deblur_bccb", "--sigma_0", str(args.sigma_0),
                "--eta", str(args.eta), "--etaB", str(args.etaB)]
    run_args, run_cfg = parse_args_and_config()
    betas = Diffusion(run_args, run_cfg, device=device).betas.to(device)

    # ------------------------------------------------------------------ #
    # 4) Load pretrained diffusion model
    # ------------------------------------------------------------------ #
    cfg = vars(run_cfg.model)
    if run_cfg.model.type.lower() == "openai":
        model = create_model(**cfg)
    else:
        model = create_model_2(**cfg).model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = torch.nn.DataParallel(model).to(device).eval()
    print("✅ Model loaded")

    # ------------------------------------------------------------------ #
    # 5) DDRM sampling
    # ------------------------------------------------------------------ #
    step_seq = list(range(0, betas.numel(), max(1, betas.numel()//args.timesteps)))
    x_start  = torch.randn_like(rf_t)

    with torch.no_grad():
        x_seq = efficient_generalized_steps(
            x_start, step_seq, model, betas,
            H_funcs, rf_t, args.sigma_0*2,
            etaA=args.eta, etaB=args.etaB, etaC=args.eta
        )

    # extract final tensor
    x_rest = x_seq
    while isinstance(x_rest, (list, tuple)):
        x_rest = x_rest[-1]
    if torch.is_complex(x_rest):
        x_rest = x_rest.real

    # map from [-1,1] back to [0,1] RF
    rf_rest = ((x_rest[:, :1] + 1)*0.5)[0, 0].cpu().numpy()

    # ------------------------------------------------------------------ #
    # 6) Save restored RF
    # ------------------------------------------------------------------ #
    out_file = Path(args.output_dir) / "restored_ddrm.mat"
    scipy.io.savemat(out_file, {"rf_restored": rf_rest})
    print(f"✅ Saved restored RF → {out_file}")


if __name__ == "__main__":
    main()
