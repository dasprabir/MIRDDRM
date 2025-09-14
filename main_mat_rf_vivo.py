import argparse
import os
import sys
import scipy.io
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import resize as resize_TF
from guided_diffusion.script_util import create_model
from functions.svd_replacement import deconvolution_BCCB
from functions.denoising import efficient_generalized_steps
from runners.diffusion import Diffusion
from main import parse_args_and_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_dir', type=str, required=True)
    parser.add_argument('--mat_list', type=str, required=True)
    parser.add_argument('--key', type=str, default='rf')
    parser.add_argument('--psf_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--timesteps', type=int, default=20)
    parser.add_argument('--eta', type=float, default=0.85)
    parser.add_argument('--etaB', type=float, default=1.0)
    parser.add_argument('--deg', type=str, default='deblur_bccb')
    parser.add_argument('--sigma_0', type=float, default=0)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='mat_output_vivo')
    parser.add_argument('--config', type=str, default='deblur_us.yml')
    parser.add_argument('--doc', type=str, default='vivo')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare config parsing via sys.argv trick
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

    # Load observed RF data
    mat_path = os.path.join(args.mat_dir, args.mat_list.strip())
    with open(mat_path, 'r') as f:
        data_name = f.readline().strip()
    rf_mat = scipy.io.loadmat(os.path.join(args.mat_dir, data_name))
    rf_observed = rf_mat[args.key]

    # Resize RF to 256x256 before DDRM
    rf_image = Image.fromarray(rf_observed.astype(np.float32))
    rf_resized = resize_TF(rf_image, (args.image_size, args.image_size), antialias=True)
    x = np.array(rf_resized).astype(np.float32)

    # Normalize RF to [-1, 1]
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x = x * 2 - 1
    x_tensor = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(torch.float32).to('cuda')

    # Expand to 3-channel to match model input
    x_tensor = x_tensor.repeat(1, 3, 1, 1)

    # Load PSF and define degradation operator
    psf_mat = scipy.io.loadmat(args.psf_path)
    psf = next((psf_mat[k] for k in psf_mat if 'psf' in k and not k.startswith('__')), None)
    if psf is None:
        raise KeyError("PSF not found in .mat file.")
    H_funcs = deconvolution_BCCB(psf, args.image_size, device='cuda')

    # Create model and diffusion instance
    diffusion = Diffusion(run_args, run_cfg, device='cuda')
    model = create_model(**vars(run_cfg.model)).to('cuda')
    model.load_state_dict(torch.load(args.model_path, map_location='cuda'))
    model.eval()

    # DDRM sampling
    sigma0 = args.sigma_0 * 2
    y0 = x_tensor
    x_init = torch.randn_like(y0)
    seq = list(range(0, diffusion.betas.shape[0],
                     max(1, diffusion.betas.shape[0] // args.timesteps)))

    with torch.no_grad():
        x_seq = efficient_generalized_steps(
            x_init, seq, model, diffusion.betas, H_funcs, y0, sigma0,
            etaA=args.eta, etaB=args.etaB, etaC=args.eta
        )

    # Handle nested list output
    restored = x_seq
    if isinstance(restored, (list, tuple)):
        restored = restored[-1]
        if isinstance(restored, (list, tuple)):
            restored = restored[-1]

    # Ensure tensor is real-valued
    if torch.is_complex(restored):
        restored = restored.real

    # Clamp and scale to [0, 1]
    x_restored = (restored.clamp(-1, 1) + 1) / 2
    restored_np = x_restored[0, 0].cpu().numpy()

    # Save result
    scipy.io.savemat(os.path.join(args.output_dir, 'restored_ddrm.mat'),
                     {'rf_restored': restored_np})
    print("✅ Saved restored RF to:", os.path.join(args.output_dir,
                                                   'restored_ddrm.mat'))

if __name__ == '__main__':
    main()
