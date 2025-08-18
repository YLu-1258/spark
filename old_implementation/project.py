import os
import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from zarr.storage import LocalStore
from tqdm.auto import tqdm
from collections import defaultdict
import yaml as yml
from ..config import *
from .dataset import EEGDataset
from ..features.extractors import extract_features
from ..preprocessing.normalization import robust_winsor_scale
from ruamel.yaml import YAML

def generate_config(project_name: str):
    """
    Generates a default configuration file in the specified project directory.
    
    Parameters:
        project_name (str): Name of the project directory.
    """
    if not os.path.exists(project_name):
        raise FileNotFoundError(f"Project directory '{project_name}' does not exist.")
    
    config = get_default_config()
    config.project.project_name = project_name
    config.paths.data_dir = os.path.join(project_name, "data")

    config_path = os.path.join(project_name, "config.yaml")
    with open(config_path, 'w') as f:
        for member in Config.__dataclass_fields__:
            f.write(f"# {member} configuration\n")
            yml.dump(asdict(getattr(config, member)), f)
            f.write("\n")
    
    print(f"Default configuration file created at {config_path}. Please review and modify as needed.")

def setup_project(project_name : str, override : bool = False):
    """
    Sets up the directory structure for a seizure detection project using GMM.
    
    Parameters:
    project_name (str): Name of the project directory to create.
    override (bool): If True, existing directories will be overridden.
    """
    if not os.path.exists(project_name):
        os.makedirs(project_name)
    elif override:
        for root, dirs, files in os.walk(project_name, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        print(f"Directory {project_name} already exists and override is set to False.")
        return
    
    # Create subdirectories
    subdirs = ['data', 'features', 'results', 'logs', 'segments']
    for subdir in subdirs:
        os.makedirs(os.path.join(project_name, subdir), exist_ok=True)

    # Generate default config file
    generate_config(project_name)

def load_config(project_name : str) -> Config:
    """
    Loads the configuration from the project's YAML file.
    
    Parameters:
    project_name (str): Name of the project directory.
    
    Returns:
    Config: Configuration object.
    """
    config_path = os.path.join(project_name, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found in {project_name}.")
    
    with open(config_path, 'r') as f:
        config = yml.safe_load(f)
    
    validate_config(config)

    return config
    
def load_data(project_name: str, config : dict):
    """
    Creates symlinks from config['original_data_dir'] to config['data_dir'] directory.
    
    Parameters:
    project_name (str): Name of the project directory.
    config (dict): Configuration dictionary.
    
    """
    data_dir = config['data_dir']
    original_data_dir = config.get('original_data_dir', None)
    
    if original_data_dir is None:
        print("No original_data_dir specified in configuration. Please manually move files into the data directory.")
        return
    
    if not os.path.exists(original_data_dir):
        raise FileNotFoundError(f"Original data directory '{original_data_dir}' does not exist.")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    for filename in os.listdir(original_data_dir):
        src = os.path.join(original_data_dir, filename)
        dst = os.path.join(data_dir, filename)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            print(f"Created symlink: {dst} -> {src}")
        else:
            print(f"File {dst} already exists. Skipping symlink creation.")
    
    print(f"Data loading complete. Files are available in {data_dir}.")

def extract_and_store_features(config: dict):
    """
    Extract features for each window of each dataset and save per-dataset feature arrays and metadata.
    """
    ds = EEGDataset(config)

    # prepare zarr store for segments
    seg_store_path = os.path.join(config['project_name'], 'segments.zarr')
    seg_store = LocalStore(seg_store_path)
    seg_root  = zarr.group(seg_store, overwrite=False)

    # Group features and metadata by dataset_id
    features_by_ds = defaultdict(list)
    meta_by_ds = defaultdict(list)

    # 1) compute total windows
    total = 0
    ws = config['window_size']
    ov = config['overlap']
    sr = config['sampling_rate']
    win_samps  = int(ws * sr)
    step_samps = int((ws - ov) * sr)

    print("calculating total number of windows...")

    total = 0
    for path in ds.raw_files:
        ds_id = Path(path).stem
        N      = ds.get_num_samples(path)
        if N < win_samps: 
            continue
        n_wins = 1 + (N - win_samps)//step_samps
        total += n_wins
        print(f"Dataset '{ds_id}': {N} samples, {n_wins} windows of {win_samps} samples each.")

        # create one big Zarr array per dataset
        seg_root.create_dataset(
        name=ds_id,
        shape=(n_wins, win_samps),
        chunks=(min( n_wins, 128 ), win_samps),  # 128 rows per chunk
        dtype='float32',
        overwrite=False
    )

    print(f"Total windows to process: {total}, Extracting...")
    for win in tqdm(ds,
                    total=total,
                    desc="Extracting windows",
                    unit="win"):
        # Extract feature vector for one window
        ds_id = win.meta.dataset_id
        w_id  = win.meta.window_id
        seg   = win.data   # 1D numpy array of length win_samps

        # Store segment in zarr
        seg_root[ds_id][win.meta.window_id, :] = seg


        fv = extract_features(win.data, config)
        features_by_ds[ds_id].append(fv)
        meta_by_ds[ds_id].append({
            'window_id': w_id,
            'start_time': win.meta.start_time,
            'end_time':   win.meta.end_time
        })

    seg_root.store.close()

    # Ensure output directory exists
    out_dir = os.path.join(config['project_name'], config['features_dir'])
    print(f"Saving features to {out_dir}...")
    os.makedirs(out_dir, exist_ok=True)
    dfs = []

    # Save each dataset's features and metadata separately
    for ds_id, feats_list in features_by_ds.items():
        # Stack feature vectors: shape (n_features, n_windows)
        feats_arr = np.stack(feats_list, axis=1)

        # robust winsorization of features
        feats_arr_padded = np.nan_to_num(feats_arr, nan=0.0)
        features_normalized = robust_winsor_scale(feats_arr_padded.T, axis=0)
        config_features = [key for key in config['feature_types']]
        config_bands = [band for band in config['osc_bands']]
        feature_names = [f"{feat}_{band}" 
                for band in config_bands
                for feat in config_features]
        features_df = pd.DataFrame(features_normalized, columns=feature_names)
        df_meta = pd.DataFrame(meta_by_ds[ds_id])
        df_all = pd.concat([df_meta.reset_index(drop=True),
                            features_df.reset_index(drop=True)], axis=1)
        df_all['dataset_id'] = ds_id
        # save one pickle per dataset
        out_path = os.path.join(out_dir, f"{ds_id}_features.pkl")
        df_all.to_pickle(out_path)
        dfs.append(df_all)

        print(f"Saved {len(df_all)} windows for '{ds_id}' → {out_path}")
    # Save all features
    all_dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
    all_out_path = os.path.join(out_dir, "all_features.pkl")
    all_dfs.to_pickle(all_out_path)
    print(f"Saved all features to {all_out_path} with {len(all_dfs)} total windows.")
    print("Feature extraction complete. With {total} windows processed.".format(total=total))
    print(f"Segments store at: {seg_store_path}")