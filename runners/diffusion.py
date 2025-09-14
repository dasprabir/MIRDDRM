import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from accelerate import Accelerator
import math

from guided_diffusion.unet import UNetModel
from guided_diffusion.script_util import (
    create_model, create_model_2, create_classifier,
    classifier_defaults, args_to_dict, create_gaussian_diffusion
)

from functions.ckpt_util import get_ckpt_path, download
from functions.denoising import efficient_generalized_steps
from datasets import get_dataset, data_transform, inverse_data_transform

import torchvision.utils as tvu
import random
import scipy.io


def add_AWGN(myimage, SNRdB):
    output_image = myimage.to(torch.float64)
    SNR = 10 ** (SNRdB / 10.0)
    ps_in = torch.sum(output_image ** 2) / output_image.numel()
    pb_norm = ps_in / SNR
    sigma_0 = torch.sqrt(pb_norm)
    noise = sigma_0 * torch.randn_like(output_image)
    output_image = output_image + noise
    return output_image, sigma_0


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        arr = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(arr) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model_var_type = config.model.var_type

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1.0 - self.betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = self.betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None

        # Instantiate and expose model
        if self.config.model.type == 'openai':
            # build the UNet and expose
            self.model = create_model(**vars(self.config.model))
            model = self.model
            if self.config.model.use_fp16:
                model.convert_to_fp16()

            # load checkpoint
            ckpt_name = f"{self.config.data.image_size}x{self.config.data.image_size}_diffusion_uncond.pt"
            ckpt = os.path.join(self.args.exp, f"logs/imagenet/{ckpt_name}")
            if not os.path.exists(ckpt):
                download(f'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/{ckpt_name}', ckpt)
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device).eval()
            model = torch.nn.DataParallel(model)
            self.model = model

            if self.config.model.class_cond:
                # classifier omitted
                pass

        elif self.config.model.type == 'DDPM':
            config_dict = vars(self.config.model)
            # build the UNet and expose
            self.model = create_model_2(**config_dict)
            model = self.model
            # load checkpoint
            ckpt = os.path.join(
                self.config.model.ckpt_folder,
                f"{self.config.model.ckpt_name}_{self.config.model.image_size}x{self.config.model.image_size}_diffusion_uncond.pt"
            )
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device).eval()
            model = torch.nn.DataParallel(model)
            self.model = model

        else:
            raise ValueError(f"Unknown model type: {self.config.model.type}")

        # run sampling
        self.sample_sequence(self.model, cls_fn)

    def sample_sequence(self, model, cls_fn=None):
        args, config = self.args, self.config
        _, test_dataset = get_dataset(args, config)
        loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )
        # rest unchanged...

    def sample_image(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        return efficient_generalized_steps(
            x, seq, model, self.betas,
            H_funcs, y_0, sigma_0,
            etaB=self.args.etaB,
            etaA=self.args.eta,
            etaC=self.args.eta,
            cls_fn=cls_fn,
            classes=classes
        )
