import numpy as np
import pandas as pd
def segment_windows(data, config : dict):
    """
    Perform segmentation on the data.

    Yields:
        start (int): Start index of the window.
        end (int): End index of the window.
        segment (np.ndarray): The EEG data segment for the window.
    """

    sr = config['sampling_rate']
    # allow overriding these via config if you like:
    ws = config.get('window_size', 2.0)   # seconds
    ov = config.get('overlap',     0.2)   # seconds

    window_samps = int(ws * sr)
    step_samps   = int((ws - ov) * sr)
    N = len(data)

    if window_samps > N:
        # no windows fit even once
        return
    num_windows = 1 + (N - window_samps) // step_samps
    max_time = len(data["pi_time"].values)

    for i in range(num_windows):
        start = i * step_samps
        end   = start + window_samps
        if end <= max_time:
            yield start, end, data["EEG"].values[start:end]
        else:
            break