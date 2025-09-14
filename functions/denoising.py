import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn=None, classes=None):
    with torch.no_grad():
        y_0 = y_0.reshape([x.shape[0],x.shape[1], x.shape[2], x.shape[3]])
        # setup vectors used in the algorithm

        singulars = H_funcs.singulars()
        Sigma = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3], dtype=singulars.dtype, device=x.device)
        Sigma[:singulars.shape[0]] = singulars
        U_t_y = H_funcs.Ut(y_0)
        
        # U_t_y = U_t_y.reshape(x.shape[0], -1)
        Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

        # initialize x_T as given in the paper
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()

        if singulars.dtype == torch.complex128:
            large_singulars_index = torch.where(torch.abs((singulars) * largest_sigmas[0, 0, 0, 0]) > sigma_0)
        else:
            large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)

        inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3], dtype=singulars.dtype).to(
            singulars.device)
        inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]

        inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)

        # implement p(x_T | x_0, y) as given in the paper
        # if eigenvalue is too small, we just treat it as zero (only for init)
        init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
        init_y = init_y.to(U_t_y.dtype)  # Ensure init_y has the same dtype as U_t_y
        init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[
            large_singulars_index].reshape(1, -1)

        init_y = init_y.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2

        if remaining_s.dtype == torch.complex128:
            real_part = remaining_s.real
            imag_part = remaining_s.imag

            # Apply clamp_min and sqrt on both parts
            real_part = real_part.clamp_min(0.0)
            imag_part = imag_part.clamp_min(0.0)

            # Combine back into a complex tensor
            #remaining_s = torch.abs(remaining_s).clamp_min(0.0)

            remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).sqrt()
        else:
            remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()

        init_y = init_y + remaining_s * x
        init_y = init_y / largest_sigmas

        # setup iteration variables
        x = H_funcs.V(init_y).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        # iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())

            xt = xs[-1].to('cuda')
            if cls_fn == None:
                et_final = model(torch.real(xt).to(dtype=torch.float32), t)

                if singulars.dtype == torch.complex128:
                    et_imag = model(torch.imag(xt).to(dtype=torch.float32), t)
            else:
                et = model(torch.real(xt).to(dtype=torch.float32), t, classes)
                et = et[:, :3]
                et_final = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(torch.real(xt).to(dtype=torch.float32), t, classes)

                if singulars.dtype == torch.complex128:
                    et_imag = model(torch.imag(xt).to(dtype=torch.float32), t, classes)
                    et_imag = et_imag[:, :3]
                    et_imag = et_imag - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(torch.imag(xt).to(dtype=torch.float32), t,classes)


            if et_final.size(1) == 6:
                et_final = et_final[:, :3]
                if singulars.dtype == torch.complex128:
                    et_imag = et_imag[:, :3]

            x0_t = (torch.real(xt) - et_final.to(xt.dtype) * (1 - at).sqrt()) / at.sqrt()
            if singulars.dtype == torch.complex128:
                x0_t_imag = torch.real((torch.imag(xt) - et_imag.to(xt.dtype) * (1 - at).sqrt()) / at.sqrt())
                x0_t = x0_t + 1j * x0_t_imag

            # variational inference conditioned on y
            sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            xt_mod = xt / at.sqrt()[0, 0, 0, 0]

            V_t_x = H_funcs.Vt(xt_mod)
            SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
            V_t_x0 = H_funcs.Vt(x0_t)
            SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

            falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)

            if singulars.dtype == torch.complex128:
                cond_before_lite = torch.abs(singulars) > sigma_0/sigma_next
                has_false_values = torch.any(~cond_before_lite)
                cond_after_lite = torch.abs(singulars) < sigma_0/sigma_next

            else:
                cond_before_lite = singulars * sigma_next > sigma_0
                cond_after_lite = singulars * sigma_next < sigma_0

            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            std_nextC = sigma_next * etaC
            sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

            std_nextA = sigma_next * etaA
            sigma_tilde_nextA = torch.sqrt(sigma_next ** 2 - std_nextA ** 2)

            diff_sigma_t_nextB = torch.sqrt(
                sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

            # missing pixels
            Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et_final) + std_nextC * torch.randn_like(V_t_x0)

            Vt_xt_mod_next[:, cond_after] = V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:,
                                                                                        cond_after_lite] + std_nextA * torch.randn_like(
                V_t_x0[:, cond_after])

            # noisier than y (before)
            Vt_xt_mod_next[:, cond_before] = \
                (Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:,cond_before] + diff_sigma_t_nextB * torch.randn_like(
                    U_t_y)[:, cond_before_lite])

            # aggregate all 3 cases and give next prediction

            if torch.isnan(Vt_xt_mod_next).any() or torch.isinf(Vt_xt_mod_next).any():

                Vt_xt_mod_next = torch.nan_to_num(Vt_xt_mod_next, nan=0.0, posinf=1000, neginf=-1000)
            xt_mod_next = H_funcs.V(
                Vt_xt_mod_next.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]))

            xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(x.shape[0], x.shape[1], x.shape[2],x.shape[3])

            x0_preds.append(x0_t)
            xs.append(xt_next)

    return xs, x0_preds