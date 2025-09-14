#!/bin/sh

#SBATCH --job-name=DDRM
#SBATCH --output=DDRM-%j.out
#SBATCH --error=DDRM-%j.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mail-type=BEGIN,END  
#SBATCH --mail-user=aghouli@irit.fr


srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif /projects/minds/aghouli-M2/mon_env/bin/python "/main.py" --ni --config imagenet_256.yml --doc imagenet_ood --timesteps 20 --eta 0.85 --etaB 1 --deg sr2 --sigma_0 0 -i imagenet_256_sigma_0




