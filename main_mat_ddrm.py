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
from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_dir', type=str, required=True)
    parser.add_argument('--mat_list', type=str, required=True)
    parser.add_argument('--key', type=str, default='rf_GT')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--psf_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--timesteps', type=int, default=20)
    parser.add_argument('--eta', type=float, default=0.85)
    parser.add_argument('--etaB', type=float, default=1.0)
    parser.add_argument('--deg', type=str, default='deblur_bccb')
    parser.add_argument('--sigma_0', type=float, default=0)
    parser.add_argument('--output_dir', type=str, default='mat_output_ddrm')
    parser.add_argument('--config', type=str, default='deblur_us.yml')
    parser.add_argument('--doc', type=str, default='mat_demo')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    dataset = get_ultrasound_mat_dataset(args.mat_dir, args.mat_list, key=args.key, image_size=args.image_size)

    sys.argv = ['main.py', '--config', args.config, '--doc', args.doc, '--ni',
                '--timesteps', str(args.timesteps), '--deg', args.deg, '--sigma_0', str(args.sigma_0),
                '--eta', str(args.eta), '--etaB', str(args.etaB), '-i', args.mat_dir]

    run_args, run_cfg = parse_args_and_config()
    cfg_model = vars(run_cfg.model)

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

    logging.info("✅ Model instantiated and checkpoint loaded.")

    psf_file = args.psf_path or os.path.join(args.mat_dir, 'psf_GT_0.mat')
    logging.info(f"Loading PSF from: {psf_file}")
    psf_mat = scipy.io.loadmat(psf_file)

    psf = None
    for key in [f'psf_ref', f'psf_GT_{args.doc}', f'psf_estim_{args.doc}']:
        if key in psf_mat:
            psf = psf_mat[key]
            logging.info(f"✅ Found PSF key: {key}")
            break
    if psf is None:
        raise KeyError("No valid PSF key found.")

    psf = psf.astype(np.float32)
    H_funcs = deconvolution_BCCB(psf, run_cfg.data.image_size, device)

    x_gray, _ = dataset[0]
    x = x_gray.unsqueeze(0).to(device)
    x = x.repeat(1, 3, 1, 1)
    x = x * 2 - 1

    sigma0 = args.sigma_0 * 2
    y0 = H_funcs.H(x)
    if torch.is_complex(y0):
        y0 = y0.real
    y0 = y0 + sigma0 * torch.randn_like(y0)

    y0_img = y0.view(*x.shape)
    if torch.is_complex(y0_img):
        y0_img = y0_img.real

    gt_np = x[0, 0].detach().cpu().numpy()
    degraded_np = y0_img[0, 0].detach().cpu().numpy()

    # Save ground truth and degraded RF
    scipy.io.savemat(os.path.join(args.output_dir, 'ground_truth.mat'), {'GT_rf_resized': gt_np})
    scipy.io.savemat(os.path.join(args.output_dir, 'degraded_y0.mat'), {'rf_degraded': degraded_np})

    # Run DDRM restoration
    T = betas.shape[0]
    skip = max(1, T // args.timesteps)
    seq = list(range(0, T, skip))
    x_init = torch.randn_like(x).to(device)

    with torch.no_grad():
        x_seq = efficient_generalized_steps(
            x_init, seq, model, betas, H_funcs, y0, sigma0,
            etaA=args.eta, etaB=args.etaB, etaC=args.eta
        )

    restored = x_seq
    if isinstance(restored, (list, tuple)):
        restored = restored[-1]
        if isinstance(restored, (list, tuple)):
            restored = restored[-1]
    if torch.is_complex(restored):
        restored = restored.real

    restored_gray = restored[:, :1, :, :]
    restored_gray = (restored_gray + 1) * 0.5
    restored_np = restored_gray[0, 0].detach().cpu().numpy()

    # Save restored RF
    scipy.io.savemat(os.path.join(args.output_dir, 'restored_ddrm.mat'), {'rf_restored': restored_np})
    logging.info(f"✅ Saved DDRM-restored RF to {args.output_dir}/restored_ddrm.mat")

if __name__ == '__main__':
    main()
