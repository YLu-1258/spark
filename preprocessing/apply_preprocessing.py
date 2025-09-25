import pandas as pd
from .filters import highpass_filter, bandpass_filter
from .normalization import robust_winsor_scale, zscore_normalize
from .padnan import fill_nan_with_zero
import numpy as np
def apply_preprocessing(data: pd.DataFrame, config : dict):
    """
    Apply preprocessing steps to the input DataFrame according to the provided configuration.
    
    Parameters:
        data (pd.DataFrame): Input DataFrame with time-series data.
        preproc_cfg: Configuration object containing preprocessing parameters.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    
    # Handle NaN values
    eeg = data['EEG'].values
    eeg_padded = fill_nan_with_zero(eeg)
    
    # High-pass filter
    eeg_filtered = highpass_filter(eeg_padded, cutoff=5, fs=2 * config['sampling_rate'])

    # robust standardization
    # Change this to zscore normalization
    # Use winsor for feature normalization
    eeg_normalized = zscore_normalize(eeg_filtered)
    return eeg_normalized
