import argparse
import logging
import os
import sys

import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2

from guided_diffusion.script_util import create_model, create_model_2
from main import parse_args_and_config
from functions.svd_replacement import deconvolution_BCCB
from functions.denoising import efficient_generalized_steps
from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)

def parse_args():
    parser = argparse.ArgumentParser(
        description='DDRM B-mode only pipeline'
    )
    parser.add_argument(
        '--mat_dir', type=str, required=True,
        help='Directory containing bmode_GT.mat'
    )
    parser.add_argument(
        '--psf_path', type=str, default=None,
        help='Optional path to PSF .mat file'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to diffusion model checkpoint'
    )
    parser.add_argument(
        '--image_size', type=int, default=256,
        help='Width & height for B-mode images'
    )
    parser.add_argument(
        '--timesteps', type=int, default=20,
        help='Number of DDRM sampling steps'
    )
    parser.add_argument(
        '--eta', type=float, default=0.85,
        help='DDR... A parameter'
    )
    parser.add_argument(
        '--etaB', type=float, default=1.0,
        help='DDR... B parameter'
    )
    parser.add_argument(
        '--deg', type=str, default='deblur_bccb',
        help='Degradation type'
    )
    parser.add_argument(
        '--sigma_0', type=float, default=0,
        help='Initial noise multiplier'
    )
    parser.add_argument(
        '--output_dir', type=str, default='mat_output',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--config', type=str, default='deblur_us.yml',
        help='Guided-diffusion config YAML'
    )
    parser.add_argument(
        '--doc', type=str, default='mat_demo',
        help='Doc tag for PSF key lookup'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # --- Load B-mode ground truth ---
    bmode_file = os.path.join(args.mat_dir, 'bmode_GT.mat')
    if not os.path.isfile(bmode_file):
        raise FileNotFoundError(f"B-mode GT file not found: {bmode_file}")
    logging.info(f"Loading B-mode GT from {bmode_file}")
    mat = scipy.io.loadmat(bmode_file)
    if 'bmode_GT' not in mat:
        raise KeyError(f"'bmode_GT' key not in {bmode_file}")
    bmode_GT = mat['bmode_GT'].astype(np.float32)
    if bmode_GT.shape != (args.image_size, args.image_size):
        bmode_GT = cv2.resize(bmode_GT, (args.image_size, args.image_size))

    # Normalize to [-1, 1] and repeat channels
    bmode_tensor = torch.from_numpy(bmode_GT / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    bmode_tensor = bmode_tensor.repeat(1, 3, 1, 1) * 2 - 1

    # Save input B-mode
    scipy.io.savemat(
        os.path.join(args.output_dir, 'bmode_input.mat'),
        {'bmode_input': bmode_GT}
    )
    plt.imsave(
        os.path.join(args.output_dir, 'bmode_input.png'),
        bmode_GT, cmap='gray'
    )

    # --- Setup diffusion runner ---
    sys.argv = [
        'main.py',
        '--config', args.config,
        '--doc',    args.doc,
        '--ni',
        '--timesteps', str(args.timesteps),
        '--deg',       args.deg,
        '--sigma_0',   str(args.sigma_0),
        '--eta',       str(args.eta),
        '--etaB',      str(args.etaB),
        '-i',          args.mat_dir,
    ]
    run_args, run_cfg = parse_args_and_config()
    cfg_model = vars(run_cfg.model)

    runner = Diffusion(run_args, run_cfg, device=device)
    betas = runner.betas.to(device)

    # Instantiate model
    if run_cfg.model.type.lower() == 'openai':
        model = create_model(**cfg_model)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        wrapper = create_model_2(**cfg_model)
        wrapper.load_state_dict(torch.load(args.model_path, map_location=device))
        model = wrapper.model
    model = torch.nn.DataParallel(model).to(device).eval()
    logging.info("✅ Model loaded")

    # --- Load PSF & build H_funcs ---
    psf_file = args.psf_path or os.path.join(args.mat_dir, 'psf_GT_0.mat')
    if not os.path.isfile(psf_file):
        raise FileNotFoundError(f"PSF file not found: {psf_file}")
    mat_psf = scipy.io.loadmat(psf_file)
    psf = None
    for key in ('psf_ref', f'psf_GT_{args.doc}', f'psf_estim_{args.doc}'):
        if key in mat_psf:
            psf = mat_psf[key].astype(np.float32)
            logging.info(f"✅ PSF key found: {key}")
            break
    if psf is None:
        raise KeyError(f"No PSF key found in {psf_file}")
    H_funcs = deconvolution_BCCB(psf, run_cfg.data.image_size, device)

    # --- Degrade B-mode ---
    sigma0 = args.sigma_0 * 2
    y0 = H_funcs.H(bmode_tensor)
    if torch.is_complex(y0):
        y0 = y0.real
    y0 = y0 + sigma0 * torch.randn_like(y0)

    degraded = ((y0[0, 0].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    scipy.io.savemat(
        os.path.join(args.output_dir, 'bmode_degraded.mat'),
        {'bmode_degraded': degraded}
    )
    plt.imsave(
        os.path.join(args.output_dir, 'bmode_degraded.png'),
        degraded, cmap='gray'
    )

    # --- DDRM restoration ---
    T = betas.shape[0]
    skip = max(1, T // args.timesteps)
    seq = list(range(0, T, skip))
    x_init = torch.randn_like(bmode_tensor).to(device)

    with torch.no_grad():
        x_seq = efficient_generalized_steps(
            x_init, seq, model, betas, H_funcs, y0, sigma0,
            etaA=args.eta, etaB=args.etaB, etaC=args.eta
        )

    restored = x_seq[-1] if isinstance(x_seq, (list, tuple)) else x_seq
    if torch.is_complex(restored):
        restored = restored.real
    restored_gray = (restored[:, :1, :, :] + 1) * 0.5
    restored_np = (restored_gray[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    scipy.io.savemat(
        os.path.join(args.output_dir, 'bmode_restored.mat'),
        {'bmode_restored': restored_np}
    )
    plt.imsave(
    os.path.join(args.output_dir, 'bmode_input.png'),
    bmode_GT,
    cmap='gray'
)

    )
    logging.info("✅ Saved restored B-mode output.")

if __name__ == '__main__':
    main()
