#!/usr/bin/env python3
"""
main_vivo_patched.py
--------------------
DDRM restoration of 512×512 ultrasound RF with a 3-channel ImageNet
diffusion model.

Key extras:
• Cosine step schedule
• Warm-start (degraded + noise)
• Light data-consistency inner loop
• Precise PSF centering via ifftshift
"""

import argparse, os, sys
from pathlib import Path
import numpy as np
import torch, scipy.io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from main import parse_args_and_config
from functions.svd_replacement import deconvolution_BCCB, unwrap_tensor   # ✱
from functions.denoising           import efficient_generalized_steps
from runners.diffusion             import Diffusion
from guided_diffusion.script_util  import create_model


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--degraded_mat', required=True)
    p.add_argument('--gt_mat',      default=None)
    p.add_argument('--psf_path',    required=True)
    p.add_argument('--model_path',  required=True)
    p.add_argument('--config',      required=True)
    p.add_argument('--timesteps',   type=int,   default=220)
    p.add_argument('--eta',         type=float, default=0.98)
    p.add_argument('--etaB',        type=float, default=1.0)
    p.add_argument('--sigma_0',     type=float, default=0.02)
    p.add_argument('--dc_iter',     type=int,   default=3)
    p.add_argument('--dc_lr',       type=float, default=0.1)
    p.add_argument('--output_dir',  default='mat_Restore_DDRM')
    p.add_argument('--doc',         default='rf_ddrm')
    return p.parse_args()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def first_numeric(d):
    for k, v in d.items():
        if not k.startswith('__') and isinstance(v, np.ndarray):
            return v.squeeze(), k
    raise KeyError('No numeric array found')


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
    out = np.zeros((H, W), dtype=np.float32)
    out[(H-ph)//2:(H-ph)//2+ph, (W-pw)//2:(W-pw)//2+pw] = psf.astype(np.float32)
    return np.fft.ifftshift(out)


def build_H_funcs(psf_path, img_shape, device, in_ch):
    d = scipy.io.loadmat(psf_path)
    psf = d.get('PSF_estim')
    if psf is None:
        psf, _ = first_numeric(d)
    psf_pad = pad_center_psf(psf.squeeze(), img_shape)

    # BCCB operator
    H = deconvolution_BCCB(psf_pad.astype(np.float32), img_shape[0], device)
    # override singulars() on the fly
    s = torch.abs(torch.fft.fft2(torch.tensor(psf_pad,
                                              dtype=torch.complex64,
                                              device=device))).flatten()
    H.singulars = lambda: s.repeat(in_ch)
    return H


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 1) degraded RF ----------------------------------------------------
    rf_deg = load_arr(args.degraded_mat, 'rf_degraded').astype(np.float32)
    H, W   = rf_deg.shape
    print('RF shape:', (H, W))

    # 2) YAML config (reuse original trainer parser) --------------------
    sys.argv = ['main.py', '--config', args.config, '--doc', args.doc,
                '--ni', '--timesteps', str(args.timesteps),
                '--deg', 'deblur_bccb', '--sigma_0', str(args.sigma_0),
                '--eta', str(args.eta), '--etaB', str(args.etaB)]
    run_args, run_cfg = parse_args_and_config()
    betas = Diffusion(run_args, run_cfg, device=device).betas.to(device)

    # 3) diffusion model (ImageNet 3-ch) --------------------------------
    model_cfg = vars(run_cfg.model)
    model_cfg['in_channels']  = 3
    model_cfg['out_channels'] = 3
    model = create_model(**model_cfg).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device),
                          strict=False)
    model = torch.nn.DataParallel(model).eval()
    print('✅ Diffusion model loaded')

    # 4) inputs ---------------------------------------------------------
    x0 = torch.tensor(rf_deg, device=device)[None, None].repeat(1, 3, 1, 1)

    # 5) degradation operator ------------------------------------------
    H_funcs = build_H_funcs(args.psf_path, (H, W), device, in_ch=3)
    print('✅ PSF operator ready')

    # 6) cosine step schedule ------------------------------------------
    T = betas.numel()
    step_seq = (0.5 * (1 - np.cos(np.linspace(0, np.pi, args.timesteps)))
                * (T-1)).round().astype(int)

    # 7) warm-start -----------------------------------------------------
    torch.manual_seed(0)
    x_start = x0 + args.sigma_0 * torch.randn_like(x0)
    
    # 8) DDRM sampling --------------------------------------------------
    with torch.no_grad():
        x_est = efficient_generalized_steps(
            x_start, step_seq, model, betas,
            H_funcs, x0, args.sigma_0 * 2,
            etaA=args.eta, etaB=args.etaB, etaC=args.eta
        )
        x_est = unwrap_tensor(x_est)  # already a Tensor

        # data-consistency loop ----------------------------------------
        for _ in range(args.dc_iter):
            B, C, H_img, W_img = x_est.shape
            y_est_flat   = H_funcs.H(x_est)
            y0_flat      = x0.view(B, -1)
            residual_flat= y_est_flat - y0_flat
            grad_flat    = H_funcs.Ht(residual_flat)
            grad_img     = grad_flat.view_as(x_est)
            x_est       -= args.dc_lr * grad_img

    # ------------------------------------------------------------------
    # 9) save & metrics
    # ------------------------------------------------------------------
    # drop any tiny imaginary residue:
    rf_rest = np.real(x_est[0, 0].cpu().numpy())

    out_path = Path(args.output_dir) / 'restored_ddrm.mat'
    scipy.io.savemat(out_path, {'rf_restored': rf_rest})
    print('✅ Saved', out_path)

    if args.gt_mat:
        rf_gt = load_arr(args.gt_mat, 'GT_rf_resized').astype(np.float32)
        psnr  = peak_signal_noise_ratio(rf_gt, rf_rest, data_range=2.0)
        ssim  = structural_similarity   (rf_gt, rf_rest, data_range=2.0)
        print(f'PSNR: {psnr:.2f} dB   SSIM: {ssim:.4f}')


if __name__ == '__main__':
    main()
