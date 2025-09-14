# MIRDDRM


**Ultrasound Image Deconvolution via Block-Circulant DDRM (BCCB-DDRM)**

This repository extends the original [MIR-DDRM](https://github.com/anesgh58/MIR-DDRM) codebase developed by **Anes Ghouli** (former intern), which demonstrated successful **deblurring of ultrasound PNG images and natural images** using a **Block-Circulant with Circulant Blocks (BCCB)** approximation within the **Denoising Diffusion Restoration Models (DDRM)** framework.

My contribution, under the supervision of **Dr. Duong-Hung Pham** and **Prof. Denis Kouamé** at the [MINDS Team, IRIT Lab](https://www.irit.fr/departement/ics/minds/), was to **adapt and extend this pipeline to work directly with raw ultrasound RF data**.

---

## 📌 Objectives & Contributions

- **Study of existing MIR-DDRM pipeline** (PNG/natural images, deblurring only).
- **Extension to ultrasound RF data** as both input and output.
- **Validation on simulated RF data**, achieving stable and promising restoration.
- **Pipeline restructuring** into two stages:
  1. **Degradation** on RF ultrasound data.  
  2. **Restoration** using the BCCB-DDRM sampling process.
- **Simulation results:** decent performance on phantoms.  
- **In vivo results:** limited performance due to challenges in **PSF estimation**.

---

## 📂 Repository Structure

```

MIRDDRM/
├─ configs/             # YAML configs for experiments
├─ functions/           # Utility functions (degradation, denoising, BCCB ops)
├─ guided\_diffusion/    # Diffusion model (U-Net, helpers)
├─ runners/             # Scripts to run DDRM sampling
├─ main\_\*.py            # Entry points for RF/B-mode restoration
├─ batch\_results/       # Experiment outputs
│   ├─ split\_simu/      # Two-stage RF degradation + DDRM restoration
│   └─ ...              # Other test batches
├─ input/               # Input datasets (mat/PNG)
├─ exp/                 # Logs, configs, experiment notes
├─ environment.yml      # Conda environment
├─ README.md
└─ .gitignore

````

---

## ⚙️ Installation

Clone the repo:
```bash
git clone https://github.com/dasprabir/MIRDDRM.git
cd MIRDDRM
````

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate mirddrm
```

Git LFS is required for large files:

```bash
git lfs install
```

---

## 🚀 Usage

### 1. Apply Degradation on RF Data

```bash
python apply_degradation.py \
  --mat_dir input/Simu/1 \
  --psf_path exp/datasets/anes_data/simu/1/psf_GT_1.mat \
  --output_dir input/Simu/1/degraded_out \
  --key GT_rf_resized \
  --image_size 256 \
  --sigma_0 0.01
```

### 2. Run DDRM Restoration

```bash
python main_mat_rf.py \
  --degraded_mat input/Simu/1/degraded_out/degraded_y0.mat \
  --gt_mat input/Simu/1/degraded_out/ground_truth.mat \
  --psf_path exp/datasets/anes_data/simu/1/psf_GT_1.mat \
  --model_path exp/logs/imagenet/256x256_diffusion_unet.pt
```

---

## 📊 Results

### Simulation Data

* RF-based BCCB-DDRM achieved **decent performance** in restoring simulated RF phantoms.
* Modular split pipeline (`degradation` + `restoration`) runs successfully.

### In Vivo Data

* Restoration quality was **limited** due to issues in **PSF estimation**.
* Indicates future research direction.

### Split Simulation Results

Located in [`batch_results/split_simu/`](batch_results/split_simu/).
Includes:

* Intermediate degradation and restoration snapshots (`exp/image_samples/...`)
* Final comparison report: [`rf_bmode_compare_Final_Result.pdf`](batch_results/split_simu/rf_bmode_compare_Final_Result.pdf)

---

## 📌 References

* **Original Repo**: [MIR-DDRM by Anes Ghouli](https://github.com/anesgh58/MIR-DDRM)
* Bahjat Kawar, Michael Elad, Stefano Ermon, Jiaming Song. *Denoising Diffusion Restoration Models*. NeurIPS 2022.
* Oleg Michailovich. *A Variational Approach to Deblurring and Denoising Ultrasound Images*. IEEE TMI, 2005.

---

## ✍️ Author

**Prabir Kumar Das**
Master’s Thesis Intern – [MINDS Team, IRIT Lab](https://www.irit.fr/departement/ics/minds/)
Université Toulouse III – Paul Sabatier
📧 [dasprabirk03@gmail.com](mailto:dasprabirk03@gmail.com)




