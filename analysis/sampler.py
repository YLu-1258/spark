import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Optional
from math import sqrt

def smart_sample(
    config: dict,
    total_windows: int,
    sample_size: Optional[int] = None,
    random_state: int = 0,
    pct: float = 0.1
) -> pd.DataFrame:
    """
    Load all feature DataFrames (pickles) and their metadata CSVs from a directory,
    cluster windows, and return a sampled subset of representative windows.

    Parameters
    ----------
    features_dir : str
        Directory containing per-dataset feature pickles named <dataset_id>_features.pkl
        and metadata CSVs named <dataset_id>_meta.csv.
    total_windows : int
        Total number of windows across all datasets (used if sample_size is None).
    sample_size : int, optional
        Number of samples (clusters) to extract. Defaults to int(sqrt(total_windows)).
    random_state : int
        Random seed for clustering.
    pct : float
        Percentage of total windows to sample if sample_size is None.

    Returns
    -------
    pd.DataFrame
        Sampled windows DataFrame containing:
          - 'dataset_id', 'window_id', 'start_time', 'end_time'
          - one column per feature (named as in the pickled DataFrames)
    """
    # Determine sample size
    if sample_size is None:
        sample_size = int(pct * total_windows)
    all_features_dir = os.path.join(config['project_name'], config['features_dir'], "all_features.pkl")
    features_dir = os.path.join(config['project_name'], config['features_dir'])
    # 1) Load and concatenate all data
    if not os.path.exists(features_dir):
        dfs = []
        for fname in os.listdir(features_dir):
            if fname.endswith('_features.pkl'):
                feat_path = os.path.join(features_dir, fname)

                # Load feature and meta
                feat_df = pd.read_pickle(feat_path)
                dfs.append(feat_df)

        all_df = pd.concat(dfs, ignore_index=True)
    else:
        all_df = pd.read_pickle(all_features_dir)
    feature_cols = [c for c in all_df.columns 
                    if c not in ('dataset_id', 'window_id', 'start_time', 'end_time')]

    # 2) Cluster on the feature columns
    X = all_df[feature_cols].values
    kmeans = KMeans(n_clusters=sample_size, random_state=random_state)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    # 3) Select the closest window in each cluster
    selected_idx = []
    for cluster_id in range(sample_size):
        inds = np.where(labels == cluster_id)[0]
        if len(inds) == 0:
            continue
        dists = np.linalg.norm(X[inds] - centers[cluster_id], axis=1)
        closest = inds[np.argmin(dists)]
        selected_idx.append(closest)

    # 4) Return the sampled DataFrame
    sampled_df = all_df.iloc[selected_idx].reset_index(drop=True)
    return sampled_df