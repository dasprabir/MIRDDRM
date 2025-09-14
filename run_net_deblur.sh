#!/bin/sh

#SBATCH --job-name=DDRM_21
#SBATCH --output=DDRM_21-%j.out
#SBATCH --error=DDRM_21-%j.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mail-type=BEGIN,END  
#SBATCH --mail-user=aghouli@irit.fr


srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif /projects/minds/aghouli-M2/mon_env/bin/python "/main.py" --ni --config deblur_us.yml --doc imagenet_ood --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_bccb --sigma_0 0.0125 -i deblur_us_sigma_0




