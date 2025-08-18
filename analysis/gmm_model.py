import numpy as np
import os
import pandas as pd
from sklearn.mixture import GaussianMixture

def fit_gmm(feature_samples, config : dict):
    """
    Fit a Gaussian Mixture Model to the provided features.

    Parameters:
    feature_samples: The feature data to fit the GMM on.
    config (dict): Configuration dictionary containing model parameters.

    Returns:
    GaussianMixture: The fitted GMM model.
    """
    if (type(feature_samples) is pd.DataFrame):
        columns = [columns for columns in feature_samples.columns if columns not in ['dataset_id', 'window_id', 'start_time', 'end_time']]
        features_feed_1st = feature_samples[columns].values
    elif (type(feature_samples) is np.ndarray):
        features_feed_1st = feature_samples
    else:
        raise ValueError("feature_samples must be either a pandas DataFrame or a numpy array.")

    print(f"Fitting GMM on features with shape: {features_feed_1st.shape}")
    if features_feed_1st.ndim != 2:
        raise ValueError("features_feed_1st must be a 2D array.")
    
    # Ensure the input is a numpy array
    features_feed_1st = np.asarray(features_feed_1st)

    # Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=config['GMM_components'], random_state=config['GMM_seed'], verbose=1)
    gmm.fit(features_feed_1st)
    return gmm

def predict_gmm(gmm, feature_samples, config : dict):
    """
    Predict the GMM cluster labels for the provided features.

    Parameters:
    gmm (GaussianMixture): The fitted GMM model.
    feature_samples (pd.DataFrame): The feature data to predict on.
    config (dict): Configuration dictionary (not used here but kept for consistency).

    Returns:
    np.ndarray: Cluster labels for each sample.
    """
    meta_columns = ['dataset_id', 'window_id', 'start_time', 'end_time']
    columns = [columns for columns in feature_samples.columns if columns not in meta_columns]
    features_feed_1st = feature_samples[columns].values
    clusters = gmm.predict(features_feed_1st)
    all_features_path = os.path.join(config['project_name'], config['features_dir'], "all_features.pkl")
    if not os.path.exists(all_features_path):
        raise FileNotFoundError(f"All features file not found at {all_features_path}. Please ensure it exists.")
    all_features_df = pd.read_pickle(all_features_path)[meta_columns]
    all_features_df['GMM_cluster'] = clusters
    return all_features_df