#!/usr/bin/env python3
"""
main_ddrm_vivo.py – DDRM on in-vivo ultrasound data (no ground truth, no synthetic degradation)
"""

import argparse
import logging
import os
import sys
import torch
import numpy as np
import scipy.io
from guided_diffusion.script_util import create_model, create_model_2
from main import parse_args_and_config
from datasets import get_ultrasound_mat_dataset
from functions.svd_replacement import deconvolution_BCCB
from functions.denoising import efficient_generalized_steps
from torchvision.utils import save_image
from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_dir',    type=str, required=True, help='Folder of .mat input files')
    parser.add_argument('--mat_list',   type=str, required=True, help='List of .mat filenames')
    parser.add_argument('--key',        type=str, default='bmode', help='Key inside .mat (e.g., "bmode")')
    parser.add_argument('--image_size', type=int, default=512, help='Expected image size')
    parser.add_argument('--psf_path',   type=str, required=True, help='Path to PSF .mat file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to DDPM model checkpoint')
    parser.add_argument('--config',     type=str, required=True, help='YAML config file')
    parser.add_argument('--doc',        type=str, default='vivo_ddrm', help='Log name / doc string')
    parser.add_argument('--timesteps',  type=int, default=20, help='Sampling steps')
    parser.add_argument('--eta',        type=float, default=0.85, help='Eta A')
    parser.add_argument('--etaB',       type=float, default=1.0, help='Eta B')
    parser.add_argument('--sigma_0',    type=float, default=0.0, help='Degradation noise sigma (set 0 for vivo)')
    parser.add_argument('--output_dir', type=str, default='mat_Restore_DDRM_vivo', help='Output folder')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load input dataset (e.g., bmode)
    dataset = get_ultrasound_mat_dataset(args.mat_dir, args.mat_list, key=args.key, image_size=args.image_size)

    # Load YAML config for model
    sys.argv = [
        'main.py',
        '--config', args.config,
        '--doc', args.doc,
        '--ni',
        '--timesteps', str(args.timesteps),
        '--deg', 'deblur_bccb',
        '--sigma_0', str(args.sigma_0),
        '--eta', str(args.eta),
        '--etaB', str(args.etaB),
        '-i', args.mat_dir
    ]
    run_args, run_cfg = parse_args_and_config()
    cfg_model = vars(run_cfg.model)

    # Load pretrained diffusion model
    runner = Diffusion(run_args, run_cfg, device=device)
    betas = runner.betas.to(device)

    model_type = run_cfg.model.type.lower()
    if model_type == 'openai':
        model = create_model(**cfg_model)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = torch.nn.DataParallel(model).to(device).eval()
    else:
        wrapper = create_model_2(**cfg_model)
        wrapper.load_state_dict(torch.load(args.model_path, map_location=device))
        model = torch.nn.DataParallel(wrapper.model).to(device).eval()

    logging.info("✅ Model loaded and ready.")

    # Load PSF
    logging.info(f"Loading PSF from: {args.psf_path}")
    psf_mat = scipy.io.loadmat(args.psf_path)
    psf = None
    for key in ['psf_ref', 'PSF_estim', f'psf_estim_{args.doc}']:
        if key in psf_mat:
            psf = psf_mat[key]
            logging.info(f"✅ PSF key found: {key}")
            break
    if psf is None:
        raise KeyError(f"No valid PSF key found in {args.psf_path}")
    H_funcs = deconvolution_BCCB(psf.astype(np.float32), run_cfg.data.image_size, device)

    # Load one input from dataset (for demo)
    x_gray, _ = dataset[0]  # shape: [1, H, W]
    x = x_gray.unsqueeze(0).to(device)             # [1, 1, H, W]
    x = x.repeat(1, 3, 1, 1)                       # [1, 3, H, W]
    x = x * 2 - 1                                  # normalize to [-1, 1]

    # In vivo: use observed (already blurred) data directly
    y0 = x.clone()
    sigma0 = args.sigma_0 * 2                      # keep 0.0 or very small for vivo

    # Sampling schedule
    T = betas.shape[0]
    skip = max(1, T // args.timesteps)
    seq = list(range(0, T, skip))
    x_init = torch.randn_like(x).to(device)

    # DDRM sampling
    with torch.no_grad():
        x_seq = efficient_generalized_steps(
            x_init, seq, model, betas,
            H_funcs, y0, sigma0,
            etaA=args.eta, etaB=args.etaB, etaC=args.eta
        )

    x_out = x_seq
    while isinstance(x_out, (list, tuple)):
        x_out = x_out[-1]
    if torch.is_complex(x_out):
        x_out = x_out.real

    restored = (x_out[:, :1, :, :] + 1) * 0.5
    restored_np = restored[0, 0].cpu().numpy()

    # Save as .mat
    out_path = os.path.join(args.output_dir, "restored_ddrm.mat")
    scipy.io.savemat(out_path, {"rf_restored": restored_np})
    logging.info(f"✅ Saved restored DDRM output to {out_path}")


if __name__ == "__main__":
    main()
