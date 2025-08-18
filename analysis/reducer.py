import umap
import pandas as pd
def reduce_features(df: pd.DataFrame, config : dict) -> pd.DataFrame:
    """
    Reduce the dimensionality of the feature set using UMAP.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and metadata.
    config : dict
        Configuration dictionary (not used in this function but included for consistency).
    Returns
    -------
    np.ndarray
        2D array of reduced features (n_samples, n_components).
    """
    columns = [columns for columns in df.columns if columns not in ['dataset_id', 'window_id', 'start_time', 'end_time']]
    features_feed_1st = df[columns].values
    reducer = umap.UMAP(n_components=config['UMAP_components'], random_state=config['UMAP_seed'], n_jobs=-1, verbose=True)
    features_umap = reducer.fit_transform(features_feed_1st)
    return features_umap