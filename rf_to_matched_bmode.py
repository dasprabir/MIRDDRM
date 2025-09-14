
import numpy as np
from scipy.signal import hilbert
from skimage.exposure import match_histograms

def rf_to_matched_bmode(RF: np.ndarray, GT_bmode: np.ndarray, increase: float = 1e-6) -> np.ndarray:
    """
    Convert RF data to B-mode and match the intensity profile to the provided ground truth B-mode image.
    
    Parameters
    ----------
    RF : np.ndarray
        3D RF data with shape (H, W, N), where N is the number of frames.
    GT_bmode : np.ndarray
        Ground truth B-mode image for histogram matching (should match shape (H, W)).
    increase : float
        Small constant to avoid log(0), used during envelope detection and log compression.

    Returns
    -------
    np.ndarray
        Histogram-matched B-mode image (uint8) with shape (H, W, N).
    """

    if RF.ndim != 3:
        raise ValueError("RF must be a 3D array (H, W, Nframes).")

    if not np.issubdtype(RF.dtype, np.floating):
        RF = RF.astype(np.float32)

    H, W, N = RF.shape
    modeB = np.empty((H, W, N), dtype=np.uint8)

    for i in range(N):
        # Envelope detection
        env = np.abs(hilbert(RF[:, :, i], axis=0))

        # Log compression (log10)
        modeB_temp = 20.0 * np.log10(env + increase)
        modeB_temp = np.clip(modeB_temp, a_min=-60, a_max=None)

        # Normalize to 0–255
        modeB_temp -= modeB_temp.min()
        max_val = modeB_temp.max()
        if max_val > 0:
            modeB_temp = 255.0 * modeB_temp / max_val

        # Histogram match to GT
        matched = match_histograms(modeB_temp, GT_bmode[:, :, 0] if GT_bmode.ndim == 3 else GT_bmode, channel_axis=None)
        modeB[:, :, i] = np.clip(matched, 0, 255).astype(np.uint8)

    return modeB
