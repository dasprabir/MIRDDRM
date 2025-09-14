import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calc_performance_metrics(reference, test):
    """
    Computes multiple performance metrics:
    - MSE, PSNR, SSIM
    - RMSE, NRMSE, MAPE
    - R-value (correlation index)
    
    Inputs:
        reference: Ground truth image (2D)
        test:      Reconstructed/restored image (2D)
    
    Returns:
        dict with all metric values
    """
    reference = reference.astype(np.float32)
    test = test.astype(np.float32)
    
    if reference.shape != test.shape:
        raise ValueError("Reference and test images must have the same dimensions.")
    
    mse = np.mean((reference - test) ** 2)
    maxI = 35 if reference.max() > 1 else 1
    psnr_val = psnr(reference, test, data_range=maxI)
    ssim_val = ssim(reference, test, data_range=maxI)
    rmse = np.sqrt(mse)
    nrmse = rmse / (reference.max() - reference.min())
    mape = np.mean(np.abs((reference - test) / (reference + 1e-8))) * 100
    r_value = 1 - np.sum((test - reference) ** 2) / np.sum(reference ** 2)
    
    return {
        "MSE": mse,
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "MAPE": mape,
        "R_value": r_value
    }
