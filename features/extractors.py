from scipy.signal import butter, filtfilt
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
feature_funcs = [
    ("Amplitude Mean", lambda x: np.mean(np.abs(x))),
    ("Variance", np.var),
    ("Skewness", skew),
    ("Kurtosis", kurtosis),
    ("ZCR", lambda x: np.count_nonzero(np.diff(np.sign(x)))),
    ("PkPk", np.ptp),
    ("NumPeaks", lambda x: find_peaks(x)[0].size)
]

bands = ["theta", "alpha", "beta", "gamma"]

def mice_osc_band_filter(data, osc_band, config, order=4):
    if "delta" in osc_band:
        lowcut = 0.5
        highcut = 4
    elif "theta" in osc_band:
        lowcut = 4
        highcut = 8
    elif "alpha" in osc_band:
        lowcut = 8
        highcut = 12
    elif "beta" in osc_band:
        lowcut = 12
        highcut = 30
    elif "gamma" in osc_band:
        lowcut = 30
        highcut = 80
    else:
        return None
    nyquist = 0.5 * config['sampling_rate']
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_features(window_data, config):
    """
    Generate time-frequency features from the input EEG data.

    Parameters:
    - window_data: 1D numpy array of EEG signal
    - config: Configuration dictionary containing 'feature_types', 'osc_bands', and 'sampling_rate'

    Returns:
    - features: 1D numpy array with features for current window
    """
    
    eeg_data = window_data
    config_feature_funcs = { key: func for key, func in feature_funcs if key in config['feature_types'] }
    config_bands = [band for band in bands if band in config['osc_bands']]
    feature_names = [f"{feat}_{band}" 
                for band in config_bands
                for feat, _ in config_feature_funcs.items() if band in config["osc_bands"] and feat in config['feature_types']]
    fv = np.zeros(len(feature_names), dtype=np.float32)
    for b, band in enumerate(config_bands):
        sig = mice_osc_band_filter(eeg_data, band, config)
        for f_idx, (_name, func) in enumerate(config_feature_funcs.items()):
            col = b * len(config_feature_funcs) + f_idx
            fv[col] = func(sig)
    return fv