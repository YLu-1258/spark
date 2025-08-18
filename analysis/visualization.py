import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd

def plot_eeg_with_clusters_from_h5(h5_path, dataset_id, cluster_labels):
    """
    Plot EEG signal and cluster labels for a dataset from HDF5.

    Parameters:
    - h5_path: str, path to HDF5 file
    - dataset_id: str, dataset identifier
    - cluster_labels: np.ndarray or pd.Series, cluster labels for each window
    """
    labels = cluster_labels[dataset_id]
    with h5py.File(h5_path, 'r') as h5f:
        # Load EEG signal and time
        eeg_signal = h5f['processed_data'][dataset_id][:]
        metadata = h5f['metadata'][dataset_id + "_window_info"][:]
        start_indices = [int(i[3]) for i in metadata]
        time_int_select = h5f['time_data'][dataset_id][:]
        start_time = time_int_select[start_indices]

    plt.ion()
    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot((time_int_select - time_int_select[0])/1000/60, eeg_signal, 'r')
    ax2.plot((start_time - time_int_select[0])/1000/60, labels, '.', markersize=5)
    [ylimA, ylimB] = ax1.get_ylim()
    plt.show()