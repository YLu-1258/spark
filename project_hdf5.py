import os
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
from IPython.display import display, clear_output
from ipywidgets import widgets
import yaml as yml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
from .config import *
from .dataset_hdf5 import HDF5EEGDataset
from .features.extractors import extract_features
from .preprocessing.normalization import robust_winsor_scale
from .analysis.gmm_model import fit_gmm, predict_gmm
import gc


def create_hdf5_dataset(config: dict, force_reload: bool = False):
    """
    Set up HDF5 data store by converting parquet files
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    force_reload : bool
        Whether to force reloading of existing data
    """
    print("🔄 Setting up HDF5 data store...")
    
    # Initialize HDF5 dataset
    ds = HDF5EEGDataset(config)
    
    # Convert parquet files to HDF5
    ds.convert_parquet_to_h5(force_reload=force_reload)
    
    print("✅ HDF5 data store setup complete")
    return ds.h5_file_path

def extract_features_hdf5(config: dict, batch_size: int = None, max_memory_gb: float = 4.0):
    """
    Extract features using HDF5 for memory-efficient data access
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    batch_size : int, optional
        Number of windows to process at once. Auto-calculated if None.
    max_memory_gb : float
        Maximum memory to use for feature processing
    """
    
    # Initialize HDF5 dataset
    ds = HDF5EEGDataset(config)
    
    # Calculate optimal batch size based on memory constraint
    if batch_size is None:
        feature_count = len(config['feature_types']) * len(config['osc_bands'])
        bytes_per_window = feature_count * 4  # float32
        max_memory_bytes = max_memory_gb * 1024**3 * 0.8  # Use 80% of available
        batch_size = min(20000, int(max_memory_bytes // bytes_per_window))
    
    print(f"Using batch size: {batch_size:,} windows")
    
    # Get dataset information
    h5_path = ds.h5_file_path
    
    with h5py.File(h5_path, 'a') as h5f:
        # Initialize features and metadata groups if they don't exist
        if 'features' not in h5f:
            h5f.create_group('features')
        if 'metadata' not in h5f:
            h5f.create_group('metadata')
        
        features_group = h5f['features']
        metadata_group = h5f['metadata']
        
        # Process each dataset
        for dataset_id in h5f['processed_data'].keys():
            print(f"\n🔄 Processing dataset: {dataset_id}")
            
            # Skip if features already exist and not forcing reload
            if dataset_id in features_group:
                print(f"Features for {dataset_id} already exist, skipping...")
                continue
            
            # Get dataset info
            dataset_info = ds.get_dataset_info(dataset_id)
            total_samples = dataset_info['total_samples']
            
            # Estimate number of windows
            sr = config['sampling_rate']
            ws = config.get('window_size', 2.0)
            ov = config.get('overlap', 0.2)
            win_samps = int(ws * sr)
            step_samps = int((ws - ov) * sr)
            
            if total_samples < win_samps:
                print(f"Dataset {dataset_id} too short for windowing, skipping...")
                continue
                
            estimated_windows = 1 + (total_samples - win_samps) // step_samps
            feature_count = len(config['feature_types']) * len(config['osc_bands'])
            
            print(f"  Estimated windows: {estimated_windows:,}")
            print(f"  Features per window: {feature_count}")
            
            # Create HDF5 datasets for features and metadata
            features_dataset = features_group.create_dataset(
                dataset_id,
                shape=(estimated_windows, feature_count),
                dtype='float32',
                chunks=True,
                compression='gzip',
                compression_opts=6
            )
            
            # Store feature names as attributes
            config_features = [key for key in config['feature_types']]
            config_bands = [band for band in config['osc_bands']]
            feature_names = [f"{feat}_{band}" 
                           for band in config_bands
                           for feat in config_features]
            
            features_dataset.attrs['feature_names'] = [name.encode('utf-8') for name in feature_names]
            
            # Create metadata dataset
            metadata_dataset = metadata_group.create_dataset(
                f"{dataset_id}_window_info",
                shape=(estimated_windows,),
                dtype=np.dtype([
                    ('window_id', 'i4'),
                    ('start_time', 'f4'),
                    ('end_time', 'f4'),
                    ('start_idx', 'i4'),
                    ('end_idx', 'i4')
                ]),
                chunks=True,
                compression='gzip',
                compression_opts=6
            )
            
            # Process windows in batches
            window_idx = 0
            feature_batch = []
            metadata_batch = []
            
            def flush_batch():
                nonlocal window_idx, feature_batch, metadata_batch
                
                if not feature_batch:
                    return
                
                batch_size_actual = len(feature_batch)
                
                # Stack features and normalize
                features_array = np.stack(feature_batch, axis=0)
                features_normalized = robust_winsor_scale(
                    features_array,
                    axis=0,
                    win_quant=(0.01, 0.99),
                    scale_quant=(0.25, 0.75)
                )
                
                # Store features in HDF5
                start_idx = window_idx - batch_size_actual
                features_dataset[start_idx:window_idx] = features_normalized
                
                # Store metadata
                metadata_array = np.array(metadata_batch, dtype=metadata_dataset.dtype)
                metadata_dataset[start_idx:window_idx] = metadata_array
                
                # Clear batch
                feature_batch.clear()
                metadata_batch.clear()
                gc.collect()
            
            # Process dataset in chunks for memory efficiency
            chunk_size_samples = 60 * 60 * sr  # 1 hour chunks
            
            for start_idx in tqdm(range(0, int(total_samples), chunk_size_samples), 
                                desc=f"Processing {dataset_id}"):
                end_idx = min(start_idx + chunk_size_samples, total_samples)
                
                # Generate windows from this chunk
                for window in ds.get_windows_from_range(dataset_id, start_idx, end_idx):
                    # Extract features
                    feature_vector = extract_features(window.data, config)
                    feature_batch.append(feature_vector)
                    
                    # Store metadata
                    metadata_batch.append((
                        window.meta.window_id,
                        window.meta.start_time,
                        window.meta.end_time,
                        int(window.meta.start_time * sr),
                        int(window.meta.end_time * sr)
                    ))
                    
                    window_idx += 1
                    
                    # Flush batch when full
                    if len(feature_batch) >= batch_size:
                        flush_batch()
            
            # Final flush
            flush_batch()
            
            # Resize datasets to actual size
            actual_windows = window_idx
            if actual_windows < estimated_windows:
                features_dataset.resize((actual_windows, feature_count))
                metadata_dataset.resize((actual_windows,))
            
            print(f"  ✅ Processed {actual_windows + 1:,} windows")
    
    print(f"\n✅ Feature extraction complete! Data stored in: {h5_path}")

    return h5_path

class HDF5FeatureLoader:
    """
    Memory-efficient feature loader that accesses HDF5 data on-demand
    """
    
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.h5_file = None  # Will be opened when needed
        
    def __enter__(self):
        self.h5_file = h5py.File(self.h5_path, 'r+')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.h5_file:
            self.h5_file.close()
    
    def get_dataset_ids(self) -> list:
        """Get all available dataset IDs"""
        if not self.h5_file:
            with h5py.File(self.h5_path, 'r+') as h5f:
                return list(h5f['features'].keys())
        return list(self.h5_file['features'].keys())
    
    def get_feature_names(self, dataset_id: str) -> list:
        """Get feature names for a dataset"""
        with h5py.File(self.h5_path, 'r+') as h5f:
            if dataset_id not in h5f['features']:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            features_dataset = h5f['features'][dataset_id]
            feature_names = [name.decode('utf-8') for name in features_dataset.attrs['feature_names']]
            return feature_names
    
    def get_features_slice(self, dataset_id: str, start_idx: int = None, end_idx: int = None) -> np.ndarray:
        """
        Get a slice of features without loading entire dataset
        
        Parameters:
        -----------
        dataset_id : str
            Dataset identifier
        start_idx : int, optional
            Starting window index. If None, starts from beginning.
        end_idx : int, optional
            Ending window index. If None, goes to end.
            
        Returns:
        --------
        np.ndarray
            Feature array with shape (n_windows, n_features)
        """
        with h5py.File(self.h5_path, 'r+') as h5f:
            if dataset_id not in h5f['features']:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            features_dataset = h5f['features'][dataset_id]
            
            if start_idx is None and end_idx is None:
                return features_dataset[:]
            elif start_idx is None:
                return features_dataset[:end_idx]
            elif end_idx is None:
                return features_dataset[start_idx:]
            else:
                return features_dataset[start_idx:end_idx]
    
    def get_metadata_slice(self, dataset_id: str, start_idx: int = None, end_idx: int = None) -> np.ndarray:
        """Get metadata slice for windows"""
        with h5py.File(self.h5_path, 'r+') as h5f:
            metadata_name = f"{dataset_id}_window_info"
            if metadata_name not in h5f['metadata']:
                raise ValueError(f"Metadata for dataset {dataset_id} not found")
            
            metadata_dataset = h5f['metadata'][metadata_name]
            
            if start_idx is None and end_idx is None:
                return metadata_dataset[:]
            elif start_idx is None:
                return metadata_dataset[:end_idx]
            elif end_idx is None:
                return metadata_dataset[start_idx:]
            else:
                return metadata_dataset[start_idx:end_idx]
    
    def get_windows_count(self, dataset_id: str) -> int:
        """Get total number of windows for a dataset"""
        with h5py.File(self.h5_path, 'r+') as h5f:
            if dataset_id not in h5f['features']:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            return h5f['features'][dataset_id].shape[0]
        
    def get_all_features(self):
        """
        Get all features from all datasets

        Returns:
        --------
        np.ndarray
            Concatenated feature array with shape (n_windows, n_features)
        """
        with h5py.File(self.h5_path, 'r+') as h5f:
            feature_lst = []
            for dataset_id in h5f['features'].keys():
                feature_lst.append(h5f['features'][dataset_id][:])
            concatenated_features = np.vstack(feature_lst)
        return concatenated_features
    
    def calculate_normalization_parameters(self, features):
        """
        Calculate normalization parameters (mean, std) for the given features.

        Returns:
        --------
        tuple
            Mean and standard deviation of the features
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)

        # store parameters in h5
        with h5py.File(self.h5_path, 'a') as h5f:
            if 'normalization_params' in h5f:
                del h5f['normalization_params']
            h5f.create_dataset('normalization_params', data=np.array([mean, std]), dtype='float32')
        return mean, std

    def get_normalization_parameters(self):
        """
        Retrieves normalization parameters (mean, std) from the HDF5 file.

        Returns:
        --------
        tuple
            Mean and standard deviation of the features
        """
        with h5py.File(self.h5_path, 'r+') as h5f:
            if 'normalization_params' not in h5f:
                raise ValueError("Normalization parameters not found")
            return h5f['normalization_params'][:]

    def normalize_all_features(self):
        """
        Normalizes all features using z-score normalization

        """
        # Not normalized yet
        normalized_features = self.get_all_features()
        try:
            mean, std = self.get_normalization_parameters()
        except Exception as e:
            print(f"Normalization parameters do not exist, Calculating...")
            mean, std = self.calculate_normalization_parameters(normalized_features)
        
        normalized_features = (normalized_features - mean) / std
        return normalized_features
    
    def normalize_features(self, features : np.ndarray) -> np.ndarray:
        """
        Normalizes passed features using z-score normalization
        """
        mean, std = self.get_normalization_parameters()
        return (features - mean) / std

    def sample_features(self, dataset_ids: list = None, sample_size: int = 100000, random_state: int = 42) -> np.ndarray:
        """
        Sample features from multiple datasets for training models
        
        Parameters:
        -----------
        dataset_ids : list, optional
            List of dataset IDs to sample from. If None, uses all datasets.
        sample_size : int
            Total number of windows to sample
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            Sampled features with shape (sample_size, n_features)
        """
        if dataset_ids is None:
            dataset_ids = self.get_dataset_ids()
        
        # Calculate total windows and sample proportionally
        total_windows = sum(self.get_windows_count(ds_id) for ds_id in dataset_ids)
        sample_size = min(sample_size, total_windows)
        
        print(f"Sampling {sample_size:,} windows from {total_windows:,} total")
        
        np.random.seed(random_state)
        sampled_features = []
        
        for ds_id in dataset_ids:
            ds_windows = self.get_windows_count(ds_id)
            ds_proportion = ds_windows / total_windows
            ds_sample_size = int(sample_size * ds_proportion)
            
            if ds_sample_size == 0:
                continue
            
            print(f"  Sampling {ds_sample_size:,} from {ds_id} ({ds_windows:,} total)")
            
            # Random sampling
            if ds_sample_size >= ds_windows:
                # Take all windows
                features = self.get_features_slice(ds_id)
            else:
                # Random sample
                indices = np.random.choice(ds_windows, ds_sample_size, replace=False)
                indices = np.sort(indices)
                
                # Load in batches to avoid memory issues
                features_list = []
                batch_size = 10000
                
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    # Convert to slices for efficiency where possible
                    if len(batch_indices) > 1 and np.all(np.diff(batch_indices) == 1):
                        # Consecutive indices - use slice
                        start_idx, end_idx = batch_indices[0], batch_indices[-1] + 1
                        batch_features = self.get_features_slice(ds_id, start_idx, end_idx)
                    else:
                        # Non-consecutive - load individually (less efficient but necessary)
                        with h5py.File(self.h5_path, 'r+') as h5f:
                            features_dataset = h5f['features'][ds_id]
                            batch_features = features_dataset[batch_indices]
                    
                    features_list.append(batch_features)
                
                features = np.vstack(features_list) if features_list else np.array([])
            
            if features.size > 0:
                sampled_features.append(features)
        
        if sampled_features:
            print("Generating all normalized samples")
            return self.normalize_features(np.vstack(sampled_features))
        else:
            return np.array([])

def create_analysis_pipeline_hdf5(h5_path: str, config: dict, output_dir: str = None):
    """
    Create analysis pipeline using HDF5 data
    
    Returns a dictionary with analysis functions that use HDF5 data on-demand
    """
    
    if output_dir is None:
        output_dir = Path(h5_path).parent / 'predictions'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating analysis pipeline with HDF5 data: {h5_path}")
    print(f"Results will be saved to: {output_dir}")
    
    def fit_gmm_on_sample(config: dict):
        """Fit GMM on sampled data"""
        n_components = config.get('GMM_components', 5)
        random_state = config.get('gmm_seed', 42)
        sample_size = config.get('gmm_sample_size', 200000)
        print("🔄 Fitting GMM on sampled data...")
        
        with HDF5FeatureLoader(h5_path) as loader:
            # Normalize features
            print("Performing feature normalization:")
            loader.normalize_all_features()

            # Sample features
            sampled_features = loader.sample_features(
                sample_size=sample_size,
                random_state=random_state
            )
            
            if sampled_features.size == 0:
                raise ValueError("No features available for GMM training")

            print(f"Training GMM on {sampled_features.shape[0]:,} samples with {n_components} components")
            
            # Fit GMM
            gmm = fit_gmm(sampled_features, config)
            
            # Save GMM model
            import joblib
            gmm_path = output_dir / 'gmm.joblib'
            joblib.dump(gmm, gmm_path)
            
            print(f"✅ GMM saved to: {gmm_path}")
            return gmm, str(gmm_path)
    
    def predict_all_gmm(gmm_path: str = None, batch_size: int = 10000):
        """Predict all windows using saved GMM, assign a label from GMM"""
        import joblib
        
        if gmm_path is None:
            gmm_path = output_dir / 'gmm.joblib'
        
        print(f"🔄 Predicting features using: {gmm_path}")
        
        # Load GMM
        gmm = joblib.load(gmm_path)
        
        with HDF5FeatureLoader(h5_path) as loader:
            dataset_ids = loader.get_dataset_ids()
            
            all_predictions = {}
            all_probabilities = {}
            
            for ds_id in tqdm(dataset_ids, desc="Predicting datasets"):
                print(f"Processing {ds_id}...")
                
                dataset_windows = loader.get_windows_count(ds_id)
                predictions = []
                probabilities = []
                
                # Process in batches
                for start_idx in range(0, dataset_windows, batch_size):
                    end_idx = min(start_idx + batch_size, dataset_windows)
                    
                    # Load batch of features
                    batch_features = loader.get_features_slice(ds_id, start_idx, end_idx)
                    normalized_features = loader.normalize_features(batch_features)
                    # Predict using GMM
                    batch_pred = gmm.predict(normalized_features)
                    batch_prob = gmm.predict_proba(normalized_features)

                    predictions.append(batch_pred)
                    probabilities.append(batch_prob)
                
                # Combine batches
                all_predictions[ds_id] = np.concatenate(predictions)
                all_probabilities[ds_id] = np.concatenate(probabilities, axis=0)
                
                print(f"  Predicted {len(all_predictions[ds_id])} windows")
            
            # Save predictions
            predictions_path = output_dir / 'gmm_predictions.npz'
            np.savez_compressed(
                predictions_path,
                **{f"{ds_id}_predictions": pred for ds_id, pred in all_predictions.items()},
                **{f"{ds_id}_probabilities": prob for ds_id, prob in all_probabilities.items()}
            )
            
            print(f"✅ Predictions saved to: {predictions_path}")
            return all_predictions, all_probabilities, str(predictions_path)
    
    def generate_clustered_dataframe(config: dict):
        h5_path = config['project_name'] + '/data_store.h5'
        npz_path = config['project_name'] + '/predictions/gmm_predictions.npz'
        if (not os.path.exists(h5_path)):
            raise ValueError(f"HDF5 data file not found: {h5_path}, run feature extraction first")
        if (not os.path.exists(npz_path)):
            raise ValueError(f"Predictions file not found: {npz_path}, run GMM prediction first")
        
        data = np.load("/Users/alexa/Projects/ChenLab/seizure_library/test_project/predictions/gmm_predictions.npz")


        all_windows = []
        with h5py.File("/Users/alexa/Projects/ChenLab/seizure_library/test_project/data_store.h5", 'a') as h5f:
            for dataset in h5f['processed_data']:
                meta_data = h5f['metadata'][dataset + '_window_info'][:]
                eeg_data = h5f['processed_data'][dataset][:]
                start_indices = [int(i[3]) for i in meta_data]
                time_int_select = h5f['time_data'][dataset][:]
                start_time = time_int_select[start_indices]
                predictions = data[dataset + '_predictions']
                df_clusters = pd.DataFrame(predictions, columns=['cluster'])
                df_meta = pd.DataFrame(meta_data)
                df_meta['dataset'] = dataset
                df_meta['start_time'] = start_time
                df_meta['cluster'] = df_clusters['cluster']

                all_windows.append(df_meta)

        df_all = pd.concat(all_windows, ignore_index=True)
        # save as file
        df_all.to_csv(config['project_name'] + '/predictions/clustered_windows.csv', index=False)
        return df_all
    
    def show_figure(cluster, config, highlight_color="tab:blue"):
        try:
            df_cluster = pd.read_csv(config['project_name'] + '/predictions/clustered_windows.csv')
        except Exception as e:
            df_cluster = generate_clustered_dataframe(config)
            print(f"Generated clustered dataframe due to error: {e}")
        # Sample then sort to improve HDF5 cache locality
        df_cluster = df_cluster[df_cluster['cluster'] == cluster]
        sample_df = df_cluster.sample(n=min(20, len(df_cluster)), random_state=42).copy()
        sample_df = sample_df.sort_values(['dataset', 'start_idx']).reset_index(drop=True)

        forward_back_time_s=config.get('window_size', 2.0)

        fig, axes = plt.subplots(5, 4, figsize=(20, 16))
        axes = axes.flatten()

        with h5py.File(h5_path, 'r') as h5f:
            ds_cache = {}

            for i, ax in enumerate(axes):
                if i >= len(sample_df):
                    ax.set_facecolor('black'); ax.axis('off'); continue

                row = sample_df.iloc[i]
                ds_name = row['dataset']

                # Lazy-open dataset handles; infer fs cheaply
                if ds_name not in ds_cache:
                    eeg_ds = h5f['processed_data'][ds_name]
                    t_ds = h5f['time_data'][ds_name]  # milliseconds
                    fs = config.get('sampling_rate', None)
                    if fs is None:
                        t2 = t_ds[0:2]
                        fs = 1000.0 / (t2[1] - t2[0]) if len(t2) == 2 and (t2[1] - t2[0]) > 0 else 500.0
                    ds_cache[ds_name] = (eeg_ds, t_ds, float(fs))

                eeg_ds, t_ds, fs = ds_cache[ds_name]
                n_samples_ds = eeg_ds.shape[0]

                # Event (chunk) indices
                ev_start = int(row['start_idx'])
                ev_end   = int(row['end_idx'])

                # Context window in samples
                pad = int(round(forward_back_time_s * fs))
                win_start = max(ev_start - pad, 0)
                win_end   = min(ev_end + pad, n_samples_ds)

                if win_end <= win_start:
                    ax.set_facecolor('lightgray')
                    ax.text(0.5, 0.5, 'Invalid range', ha='center', va='center', fontsize=10)
                    ax.axis('off')
                    continue

                # Read ONLY the needed slices
                t_win_ms = t_ds[win_start:win_end]
                y = eeg_ds[win_start:win_end]

                # Anchor time to the event start so the event spans [0, duration]
                t0_ms = t_ds[ev_start]

                if ev_end < n_samples_ds:
                    t1_ms = t_ds[ev_end]
                else:
                    t1_ms = t_ds[ev_end - 1] + (1000.0 / fs)

                x = (t_win_ms - t0_ms) / 1000.0

                ax.plot(x, y, linewidth=0.8)

                # Blue highlight exactly over the sampled chunk
                chunk_duration_s = (t1_ms - t0_ms) / 1000.0
                ax.axvspan(0.0, chunk_duration_s, facecolor=highlight_color, alpha=0.3)

                # Ensure requested context is visible on both sides (may be truncated at edges)
                ax.set_xlim(-forward_back_time_s, chunk_duration_s + forward_back_time_s)

                # Robust y-limits
                lo, hi = np.percentile(y, [1, 99])
                margin = max((hi - lo) * 0.1, 1e-6)
                ax.set_ylim(lo - margin, hi + margin)

                ax.set_title(f"Sample {i+1}")
                ax.axis('off')

        fig.suptitle(f"Cluster {cluster} - Random Samples", fontsize=20)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
    
    def interactive_label_clusters_ui(config):
        try:
            df_cluster = pd.read_csv(config['project_name'] + '/predictions/clustered_windows.csv')
        except Exception as e:
            df_cluster = generate_clustered_dataframe(config)
            print(f"Generated clustered dataframe due to error: {e}")

        output_folder = Path(config.get("project_name", None))
        clusters = sorted(df_cluster['cluster'].unique())
        cluster_labels = [None] * len(clusters)
        figures = [None] * len(clusters)

        pdf_path = output_folder / "results" / "example_clustered_signals.pdf"
        csv_path = output_folder / "results" / "labels.csv"
        label_options = config.get("label_options", None)
        if label_options is None:
            label_options = ["seizure", "normal", "artifact", "unknown"]

        current_index = {'value': 0}
        selected_label_index = [None]
        
        # Widgets
        output = widgets.Output()
        label_buttons = []
        label_box = widgets.VBox()
        next_button = widgets.Button(description="Next ▶", button_style='success')
        back_button = widgets.Button(description="◀ Back", button_style='warning')
        select_cluster = widgets.Dropdown(options=clusters, description="Select cluster")
        status_label = widgets.Label()
        control_box = widgets.HBox([back_button, next_button, select_cluster])

        def render_label_buttons(selected=None):
            label_buttons.clear()
            label_box.children = []
            for i, label in enumerate(label_options):
                marker = "✅ " if i == selected else "⬜ "
                btn = widgets.Button(description=marker + label, layout=widgets.Layout(width='150px'))
                btn.on_click(lambda b, idx=i: select_label(idx))
                label_buttons.append(btn)
            label_box.children = label_buttons

        def select_label(index):
            selected_label_index[0] = index
            render_label_buttons(selected=index)

        def display_samples():
            idx = current_index['value']
            cluster = clusters[idx]

            fig = show_figure(cluster, config)
            figures[idx] = fig

            # Restore previous label if any
            prev_label = cluster_labels[idx][1] if cluster_labels[idx] else None
            selected_idx = label_options.index(prev_label) if prev_label in label_options else None
            selected_label_index[0] = selected_idx
            render_label_buttons(selected_label_index[0])
            status_label.value = f"Cluster {cluster} ({idx + 1}/{len(clusters)})"

            with output:
                clear_output(wait=True)
                display(widgets.HBox([label_box]))
                display(fig)
            plt.close(fig)
            return None

        def on_next(b):
            idx = current_index['value']
            if selected_label_index[0] is None:
                status_label.value = "⚠️ Please select a label before continuing."
                return
            cluster_labels[idx] = (clusters[idx], label_options[selected_label_index[0]])
            if idx < len(clusters) - 1:
                current_index['value'] += 1
                display_samples()
            else:
                # Save PDF
                with PdfPages(pdf_path) as pdf:
                    for fig in figures:
                        pdf.savefig(fig)
                        plt.close(fig)
                # Save labels CSV
                
                label_df = pd.DataFrame(cluster_labels, columns=["cluster", "label"])
                label_df.to_csv(csv_path, index=False)
                # edit clustered_windows here
                windows_path = config.get("project_name", None) + "/predictions/clustered_windows.csv"
                windows_df = pd.read_csv(windows_path)
                windows_df = windows_df.merge(
                    label_df,
                    on='cluster', how='left'
                )
                windows_df.to_csv(windows_path)
                with output:
                    clear_output()
                    print(f"\n✅ All done!")
                    print(f"📄 PDF saved to: {pdf_path}")
                    print(f"📝 Labels saved to: {csv_path}")
                    display(label_df)

        def on_back(b):
            if current_index['value'] > 0:
                current_index['value'] -= 1
                display_samples()
                
        def on_select(b):
            idx = clusters.index(select_cluster.value)
            current_index['value'] = idx
            display_samples()

        next_button.on_click(on_next)
        back_button.on_click(on_back)
        select_cluster.observe(on_select, names='value')

        # Layout
        display(widgets.VBox([status_label, control_box, output]))
        display_samples()

        
    def create_figure_pdf(config, pdf_name="cluster_samples.pdf"):
        try:
            df_cluster = pd.read_csv(config['project_name'] + '/predictions/clustered_windows.csv')
        except Exception as e:
            df_cluster = generate_clustered_dataframe(config)
            print(f"Generated clustered dataframe due to error: {e}")

        
        all_clusters = df_cluster['cluster'].unique()
        all_clusters.sort()
        output_dir = Path(config['project_name']) / 'predictions'
        pdf_path = output_dir / pdf_name
        with PdfPages(pdf_path) as pdf:
            for cluster in all_clusters:
                fig = show_figure(cluster, config)
                pdf.savefig(fig)
                plt.close(fig)
        
        print(f"✅ Saved cluster samples to PDF: {pdf_path}")
        return pdf_path
    
    return {
        'h5_path': h5_path,
        'output_dir': str(output_dir),
        'fit_gmm': fit_gmm_on_sample,
        'predict_gmm': predict_all_gmm,
        'create_figure_pdf': create_figure_pdf,
        'create': show_figure,
        'label': interactive_label_clusters_ui,
        'loader_class': HDF5FeatureLoader
    }