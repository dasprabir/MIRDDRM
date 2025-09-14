import argparse, logging, os, sys
import torch, numpy as np, scipy.io
from guided_diffusion.script_util import create_model, create_model_2
from main import parse_args_and_config
from datasets import get_ultrasound_mat_dataset
from functions.svd_replacement import deconvolution_BCCB
from functions.denoising import efficient_generalized_steps
from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mat_dir',    required=True)
    p.add_argument('--mat_list',   required=True)
    p.add_argument('--key',        default='rf')
    p.add_argument('--image_size', type=int, default=256)

    p.add_argument('--psf_path',   default=None)
    p.add_argument('--model_path', required=True)

    p.add_argument('--timesteps',  type=int, nargs='+', default=[20, 50, 100])
    p.add_argument('--eta',  type=float, default=0.85)
    p.add_argument('--etaB', type=float, default=1.0)
    p.add_argument('--deg',  default='deblur_bccb')
    p.add_argument('--sigma_0', type=float, default=0)

    p.add_argument('--output_dir', default='compare_timesteps')
    p.add_argument('--config',     default='deblur_us.yml')
    p.add_argument('--doc',        default='vivo')
    return p.parse_args()


# ------------------------------------------------------------------
def run_ddrm(t, *, args, device, degraded_np, y0, model, betas, H_funcs):
    skip = max(1, betas.shape[0] // t)
    seq  = list(range(0, betas.shape[0], skip))
    x_init = torch.randn_like(y0)

    with torch.no_grad():
        x_seq = efficient_generalized_steps(
            x_init, seq, model, betas, H_funcs, y0,
            sigma_0=args.sigma_0 * 2,
            etaA=args.eta, etaB=args.etaB, etaC=args.eta
        )

    # --- unwrap nested list / tuple ---
    x_rest = x_seq
    while isinstance(x_rest, (list, tuple)):
        x_rest = x_rest[-1]
    # -----------------------------------

    if torch.is_complex(x_rest):
        x_rest = x_rest.real

    x_rest = (x_rest[:, :1] + 1) * 0.5
    rf_rest = x_rest[0, 0].cpu().numpy()
    rf_rest = np.clip(rf_rest, 0, 1)
    rf_rest = rf_rest * (degraded_np.max() - degraded_np.min()) + degraded_np.min()

    out_path = os.path.join(args.output_dir, f'restored_t{t}.mat')
    scipy.io.savemat(out_path, {'rf_restored': rf_rest})
    logging.info(f'✅ saved {out_path}')


# ------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # load RF (and ensure square) ----------------------------------
    ds = get_ultrasound_mat_dataset(args.mat_dir, args.mat_list,
                                    key=args.key, image_size=args.image_size)
    x_raw, _ = ds[0]
    if x_raw.dim() == 1:
        side = int(np.sqrt(x_raw.numel()))
        x_raw = x_raw.view(side, side)
    elif x_raw.dim() == 3:
        x_raw = x_raw[0]

    x = x_raw.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)  # [1,3,H,W]
    x = x * 2 - 1
    y0 = x.clone()

    degraded_np = y0[0, 0].cpu().numpy().copy()
    scipy.io.savemat(os.path.join(args.output_dir, 'degraded_rf.mat'),
                     {'rf_input': degraded_np})

    # diffusion setup ----------------------------------------------
    sys.argv = ['main.py','--config',args.config,'--doc',args.doc,'--ni',
                '--timesteps', str(args.timesteps[-1]), '--deg', args.deg,
                '--sigma_0', str(args.sigma_0),'--eta',str(args.eta),
                '--etaB',str(args.etaB),'-i',args.mat_dir]
    run_args, run_cfg = parse_args_and_config()
    betas = Diffusion(run_args, run_cfg, device=device).betas.to(device)

    # load model ----------------------------------------------------
    mdl_cfg = vars(run_cfg.model)
    if run_cfg.model.type.lower() == 'openai':
        model = create_model(**mdl_cfg)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = torch.nn.DataParallel(model).to(device).eval()
    else:
        wrap = create_model_2(**mdl_cfg)
        wrap.load_state_dict(torch.load(args.model_path, map_location=device))
        model = torch.nn.DataParallel(wrap.model).to(device).eval()
    logging.info('✅ model checkpoint loaded')

    # PSF → operator -----------------------------------------------
    psf_path = args.psf_path or os.path.join(args.mat_dir,'psf_estim_vivo.mat')
    psf_mat  = scipy.io.loadmat(psf_path)
    psf      = next((psf_mat[k] for k in ['psf_ref',f'psf_GT_{args.doc}',
                   f'psf_estim_{args.doc}','PSF_estim'] if k in psf_mat), None)
    if psf is None: raise KeyError('No valid PSF key in psf file')
    H_funcs = deconvolution_BCCB(psf.astype(np.float32),
                                 run_cfg.data.image_size, device)

    # run for each timestep count ----------------------------------
    for t in args.timesteps:
        run_ddrm(t, args=args, device=device,
                 degraded_np=degraded_np, y0=y0,
                 model=model, betas=betas, H_funcs=H_funcs)

if __name__ == '__main__':
    main()
