import argparse
import logging
import os
import sys
import torch
import torch.nn.functional as F
from guided_diffusion.script_util import create_model, create_model_2
from main import parse_args_and_config
from datasets import get_ultrasound_mat_dataset
from functions.denoising import efficient_generalized_steps
from torchvision.utils import save_image
from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mat_dir',    type=str,   required=True, help='Path to .mat files')
    p.add_argument('--mat_list',   type=str,   required=True, help='Path to mat_list.txt')
    p.add_argument('--key',        type=str,   default='bmode_GT',  help='Key in .mat file')
    p.add_argument('--image_size', type=int,   default=256,        help='Target HR image size')
    p.add_argument('--model_path', type=str,   required=True,      help='Diffusion checkpoint')
    p.add_argument('--timesteps',  type=int,   default=20,         help='Number of sampling steps')
    p.add_argument('--eta',        type=float, default=0.85,       help='Noise scale η')
    p.add_argument('--etaB',       type=float, default=1.0,        help='Measurement noise ηB')
    p.add_argument('--sigma_0',    type=float, default=0.0,        help='Initial σ₀')
    p.add_argument('--output_dir', type=str,   default='mat_output',   help='Output folder')
    p.add_argument('--config',     type=str,   default='sr_us.yml',    help='YAML config')
    p.add_argument('--doc',        type=str,   default='sr_demo',      help='Logging doc string')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"[SR] Device: {device}")

    # load LR dataset
    scale = 4
    lr_size = args.image_size // scale
    dataset = get_ultrasound_mat_dataset(
        args.mat_dir, args.mat_list, key=args.key, image_size=lr_size
    )

    # diffusion config
    sys.argv = [
        'main.py', '--config', args.config, '--doc', args.doc,
        '--ni', '--timesteps', str(args.timesteps),
        '--deg', 'identity',  # use identity measurement
        '--sigma_0', str(args.sigma_0), '--eta', str(args.eta),
        '--etaB', str(args.etaB), '-i', args.mat_dir
    ]
    run_args, run_cfg = parse_args_and_config()
    cfg_model = vars(run_cfg.model)
    runner = Diffusion(run_args, run_cfg, device=device)
    betas = runner.betas.to(device)

    # load model
    if run_cfg.model.type.lower() == 'openai':
        net = create_model(**cfg_model)
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        model = torch.nn.DataParallel(net).to(device).eval()
    else:
        wrap = create_model_2(**cfg_model)
        wrap.load_state_dict(torch.load(args.model_path, map_location=device))
        model = torch.nn.DataParallel(wrap.model).to(device).eval()
    logging.info("✅ Model loaded")

    # prepare x: LR -> bicubic up -> normalize
    x_gray, _ = dataset[0]
    x_lr = x_gray.unsqueeze(0).to(device)
    x_up = F.interpolate(x_lr, scale_factor=scale, mode='bicubic', align_corners=False)
    x = (x_up.repeat(1, 3, 1, 1) * 2) - 1

    # identity measurement: just add noise
    sigma0 = args.sigma_0 * 2
    y0 = x + sigma0 * torch.randn_like(x)

    # save inputs
    save_image((y0 + 1) * 0.5, os.path.join(args.output_dir, 'degraded_y0.png'))
    save_image((x[:, :1] + 1) * 0.5, os.path.join(args.output_dir, 'ground_truth.png'))

    # DDRM sampling
    T = betas.shape[0]
    skip = max(1, T // args.timesteps)
    seq = list(range(0, T, skip))
    x_init = torch.randn_like(x).to(device)
    with torch.no_grad():
        x_seq = efficient_generalized_steps(
            x_init, seq, model, betas,
            None,   # no degradation operator
            y0, sigma0,
            etaA=args.eta, etaB=args.etaB, etaC=args.eta
        )

    # save restored
    out = x_seq[-1] if isinstance(x_seq, (list, tuple)) else x_seq
    out = out.real if torch.is_complex(out) else out
    restored = (out[:, :1, :, :] + 1) * 0.5
    save_image(restored, os.path.join(args.output_dir, 'restored_ddrm.png'))
    logging.info(f"✅ Saved SR-DDRM output to {args.output_dir}/restored_ddrm.png")

if __name__ == '__main__':
    main()