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
    parser.add_argument('--mat_dir',    type=str,   required=True, help='Path to .mat files')
    parser.add_argument('--mat_list',   type=str,   required=True, help='Path to mat_list.txt')
    parser.add_argument('--key',        type=str,   default='bmode_GT', help='Key in .mat file')
    parser.add_argument('--image_size', type=int,   default=256,     help='Resize dimension')
    parser.add_argument('--psf_path',   type=str,   default=None,    help='Optional path to psf_GT_*.mat')
    parser.add_argument('--model_path', type=str,   required=True, help='Path to diffusion checkpoint')
    parser.add_argument('--timesteps',  type=int,   default=20,      help='Number of sampling steps')
    parser.add_argument('--eta',        type=float, default=0.85,    help='Noise scale eta')
    parser.add_argument('--etaB',       type=float, default=1.0,     help='Measurement noise scale etaB')
    parser.add_argument('--deg',        type=str,   default='deblur_bccb', help='Degradation type')
    parser.add_argument('--sigma_0',    type=float, default=0,       help='Initial noise sigma')
    parser.add_argument('--output_dir', type=str,   default='mat_output', help='Output folder')
    parser.add_argument('--config',     type=str,   default='deblur_us.yml', help='YAML config')
    parser.add_argument('--doc',        type=str,   default='mat_demo', help='Runner doc string')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load dataset
    dataset = get_ultrasound_mat_dataset(args.mat_dir, args.mat_list, key=args.key, image_size=args.image_size)

    # Prepare diffusion config
    sys.argv = [
        'main.py',
        '--config', args.config,
        '--doc', args.doc,
        '--ni',
        '--timesteps', str(args.timesteps),
        '--deg', args.deg,
        '--sigma_0', str(args.sigma_0),
        '--eta', str(args.eta),
        '--etaB', str(args.etaB),
        '-i', args.mat_dir
    ]
    run_args, run_cfg = parse_args_and_config()
    cfg_model = vars(run_cfg.model)

    # Create beta schedule
    runner = Diffusion(run_args, run_cfg, device=device)
    betas = runner.betas.to(device)

    # Load pretrained model
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

    # Load PSF and construct operator
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
        raise KeyError(
            f"No valid PSF key found in {psf_file}. Tried: psf_ref, psf_GT_{args.doc}, psf_estim_{args.doc}"
        )

    psf = psf.astype(np.float32)
    H_funcs = deconvolution_BCCB(psf, run_cfg.data.image_size, device)

    # Load image and prepare
    x_gray, _ = dataset[0]
    x = x_gray.unsqueeze(0).to(device)
    x = x.repeat(1, 3, 1, 1)  # to 3-channel
    x = x * 2 - 1             # normalize to [-1, 1]

    # Simulate degraded measurement
    sigma0 = args.sigma_0 * 2
    y0 = H_funcs.H(x)
    if torch.is_complex(y0):
        y0 = y0.real
    y0 = y0 + sigma0 * torch.randn_like(y0)

    B, C, H, W = x.shape
    y0_img = y0.view(B, C, H, W)
    if torch.is_complex(y0_img):
        y0_img = y0_img.real

    # Save degraded and GT images as .mat
    degraded_np = y0_img[0, 0].detach().cpu().numpy()
    gt_np = x[0, 0].detach().cpu().numpy()
    scipy.io.savemat(os.path.join(args.output_dir, 'degraded_y0.mat'), {'rf_degraded': degraded_np})
    scipy.io.savemat(os.path.join(args.output_dir, 'ground_truth.mat'), {'GT_rf_resized': gt_np})

    # Sampling schedule
    T = betas.shape[0]
    skip = max(1, T // args.timesteps)
    seq = list(range(0, T, skip))
    x_init = torch.randn_like(x).to(device)

    # Run DDRM
    with torch.no_grad():
        x_seq = efficient_generalized_steps(
            x_init, seq, model, betas,
            H_funcs, y0, sigma0,
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

    # Save restored image as .mat
    restored_np = restored_gray[0, 0].detach().cpu().numpy()
    scipy.io.savemat(os.path.join(args.output_dir, 'restored_ddrm.mat'), {'rf_restored': restored_np})

    logging.info(f"✅ Saved DDRM-restored image to {args.output_dir}/restored_ddrm.mat")

    # --- Convert RFs to B-mode ---
    from rf_to_matched_bmode import rf_to_matched_bmode
    import matplotlib.pyplot as plt

    # Prepare 3D shapes (H, W, N)
    def make_3d(arr):
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        return np.transpose(arr, (1, 2, 0))

    GT_rf_3d = make_3d(gt_np)
    degraded_3d = make_3d(degraded_np)
    restored_3d = make_3d(restored_np)

    # Create B-mode images from RF
    bmode_GT = rf_to_matched_bmode(GT_rf_3d, GT_rf_3d)
    bmode_degraded = rf_to_matched_bmode(degraded_3d, GT_rf_3d)
    bmode_restored = rf_to_matched_bmode(restored_3d, GT_rf_3d)

    # Save B-mode .mat files
    scipy.io.savemat(os.path.join(args.output_dir, 'bmode_GT.mat'), {'bmode_GT': bmode_GT})
    scipy.io.savemat(os.path.join(args.output_dir, 'bmode_degraded.mat'), {'bmode_degraded': bmode_degraded})
    scipy.io.savemat(os.path.join(args.output_dir, 'bmode_restored.mat'), {'bmode_restored': bmode_restored})

    # Save RF and B-mode PNG images
    plt.imsave(os.path.join(args.output_dir, 'GT_rf.png'), GT_rf_3d[:, :, 0], cmap='gray')
    plt.imsave(os.path.join(args.output_dir, 'rf_degraded.png'), degraded_3d[:, :, 0], cmap='gray')
    plt.imsave(os.path.join(args.output_dir, 'rf_restored.png'), restored_3d[:, :, 0], cmap='gray')

    plt.imsave(os.path.join(args.output_dir, 'bmode_GT.png'), bmode_GT[:, :, 0], cmap='gray')
    plt.imsave(os.path.join(args.output_dir, 'bmode_degraded.png'), bmode_degraded[:, :, 0], cmap='gray')
    plt.imsave(os.path.join(args.output_dir, 'bmode_restored.png'), bmode_restored[:, :, 0], cmap='gray')


if __name__ == '__main__':
    main()
