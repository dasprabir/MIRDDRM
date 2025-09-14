from scipy.signal import hilbert
import numpy as np

def rf_to_bmode_morin_style(rf, increase=1.0):
    """
    Reproduces Renaud Morin's rf2bmode.m logic.
    Converts an RF image into a B-mode image using Hilbert envelope + log compression + linear scaling.
    
    Args:
        rf (ndarray): 2D RF data.
        increase (float): Small positive constant added inside log to avoid singularity.
    
    Returns:
        bmode (ndarray): 8-bit B-mode image.
    """
    log_env = 20.0 * np.log10(np.abs(hilbert(rf, axis=0)) + increase)
    log_env -= log_env.min()
    log_env *= 255.0 / log_env.max()
    return np.clip(log_env, 0, 255).astype(np.uint8)
