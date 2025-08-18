import pickle
import os
import pandas as pd
def get_all_features(config: dict) -> pd.DataFrame:
    """
    Load and concatenate all per-dataset feature DataFrames from disk.

    Parameters
    ----------
    config : dict
        Must contain:
          - 'project_name': root project directory
          - 'features_dir' : subdirectory under project_name where '<dataset>_data.pkl' lives

    Returns
    -------
    pd.DataFrame
        A single DataFrame with all windows' metadata and features concatenated.
        If no feature files are found, returns an empty DataFrame.
    """
    # Construct full path to the features directory
    features_dir = os.path.join(config['project_name'], config['features_dir'])
    all_features_path = os.path.join(features_dir, "all_features.pkl")
    dfs = []
    if not os.path.exists(all_features_path):
        return pd.read_pickle(all_features_path)
    # Check if all features file exists
    if os.path.exists(all_features_path):
        return pd.read_pickle(all_features_path)
    else:
        # Iterate over each pickle file in the directory
        for filename in os.listdir(features_dir):
            if not filename.endswith('_features.pkl'):
                continue
        path = os.path.join(features_dir, filename)
        # Use pandas to read the pickled DataFrame
        df = pd.read_pickle(path)
        dfs.append(df)

    # Concatenate all DataFrames, or return empty if none
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    config = {
        'project_name': 'test_project',
        'features_dir': 'features'
    }
    all_features = get_all_features(config)
    print(all_features.head())
    print(f"Total features extracted: {len(all_features)}")