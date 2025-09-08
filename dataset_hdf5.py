import os
import numpy as np
import pandas as pd
import h5py
import duckdb
from pathlib import Path
from .preprocessing.apply_preprocessing import apply_preprocessing
from .preprocessing.segment_windows import segment_windows
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

@dataclass
class WindowMeta:
    dataset_id: str
    window_id: int
    start_time: float
    end_time: float

@dataclass
class Window:
    meta: WindowMeta
    data: np.ndarray 

class HDF5EEGDataset:
    """
    Memory-efficient streaming dataset that processes data in chunks
    and stores everything in HDF5 format for on-demand access
    """
    
    def __init__(self, config, chunk_size_minutes: int = 60, h5_file_path: str = None):
        print("DATA DIR:", config['data_dir'])
        self.raw_files = self.list_raw_files(config['data_dir'])
        self.config = config
        self.chunk_size_minutes = chunk_size_minutes
        self.chunk_size_ms = chunk_size_minutes * 60 * 1000
        
        # Set up HDF5 file path
        if h5_file_path is None:
            h5_file_path = os.path.join(config['project_name'], 'data_store.h5')
        self.h5_file_path = h5_file_path
        
        # Initialize HDF5 file
        self._initialize_h5_structure()
        
    def list_raw_files(self, directory_path, raw_extensions=['.parquet', '.csv', '.xlsx']):
        raw_files = []
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in raw_extensions):
                raw_files.append(os.path.join(directory_path, filename))
        return raw_files
    
    def _initialize_h5_structure(self):
        """Initialize HDF5 file structure if it doesn't exist"""
        if os.path.exists(self.h5_file_path):
            print(f"Using existing HDF5 file: {self.h5_file_path}")
            return
            
        print(f"Creating HDF5 data store: {self.h5_file_path}")
        os.makedirs(os.path.dirname(self.h5_file_path), exist_ok=True)
        
        with h5py.File(self.h5_file_path, 'w') as h5f:
            # Only create processed_data, features, and metadata groups
            h5f.create_group('processed_data') # Preprocessed EEG data  
            h5f.create_group('time_data')
            h5f.create_group('features')       # Extracted features (includes window metadata)
            h5f.create_group('metadata')       # Dataset-level metadata
            
    def convert_parquet_to_h5(self, force_reload: bool = False):
        """
        Convert all parquet files to HDF5 format for efficient access
        This only needs to be done once per dataset
        """
        with h5py.File(self.h5_file_path, 'a') as h5f:
            for path in self.raw_files:
                dataset_id = Path(path).stem
                # Skip if already converted and not forcing reload
                if dataset_id in h5f['processed_data'] and not force_reload:
                    print(f"Dataset {dataset_id} already converted, skipping...")
                    continue
                print(f"Converting {dataset_id} to HDF5 (preprocessed only)...")
                # Get file info using DuckDB
                con = duckdb.connect(database=':memory:')
                file_info = con.execute(f"""
                    SELECT MIN(pi_time) as min_time, MAX(pi_time) as max_time, COUNT(*) as total_samples
                    FROM parquet_scan('{path}')
                """).fetchdf().iloc[0]
                con.close()
                min_time, max_time = file_info['min_time'], file_info['max_time']
                total_samples = file_info['total_samples']
                print(f"  Total samples in Dataset: {total_samples:,}")
                print(f"  Duration of Dataset: {(max_time - min_time)/1000/3600:.1f} hours")
                # Single-pass: create extendable HDF5 dataset and append chunks as processed
                processed_dataset = h5f['processed_data'].create_dataset(
                    dataset_id,
                    shape=(0,),
                    maxshape=(None,),
                    dtype='float32',
                    chunks=True,
                    compression='gzip',
                    compression_opts=6
                )
                processed_dataset_time = h5f['time_data'].create_dataset(
                    dataset_id,
                    shape=(0,),
                    maxshape=(None,),
                    dtype='float32',
                    chunks=True,
                    compression='gzip',
                    compression_opts=6
                )
                h5f['processed_data'][dataset_id].attrs['min_time'] = min_time
                h5f['processed_data'][dataset_id].attrs['max_time'] = max_time
                h5f['processed_data'][dataset_id].attrs['total_samples'] = total_samples
                h5f['processed_data'][dataset_id].attrs['source_file'] = path
                current_time = min_time
                chunk_idx = 0
                while current_time < max_time:
                    chunk_end = min(current_time + self.chunk_size_ms, max_time)
                    chunk_df = self._load_chunk_duckdb(path, current_time, chunk_end)
                    if chunk_df.empty:
                        current_time = chunk_end
                        continue
                    eeg_data = chunk_df['EEG'].values
                    time_data = chunk_df['Time'].values
                    preprocessed = apply_preprocessing(pd.DataFrame({'EEG': eeg_data}), self.config)
                    chunk_size = len(preprocessed)
                    # Resize dataset to accommodate new chunk
                    old_size = processed_dataset.shape[0]
                    old_size_time = processed_dataset_time.shape[0]
                    processed_dataset.resize((old_size + chunk_size,))
                    processed_dataset_time.resize((old_size_time + chunk_size,))
                    processed_dataset[old_size:old_size + chunk_size] = preprocessed
                    processed_dataset_time[old_size_time:old_size_time + chunk_size] = time_data
                    print(f"  Chunk {chunk_idx}: {len(chunk_df):,} samples (preprocessed: {chunk_size})")
                    current_time = chunk_end
                    chunk_idx += 1
                    del chunk_df
                print(f"  ✅ {dataset_id} converted and preprocessed with {chunk_idx} chunks")
    
    def _load_chunk_duckdb(self, path: str, start_time: float, end_time: float) -> pd.DataFrame:
        """Load a time chunk from parquet file using DuckDB"""
        con = duckdb.connect(database=':memory:')
        
        try:
            con.execute(f"""
                CREATE TABLE temp_data AS
                SELECT *, (pi_time - FIRST_VALUE(pi_time) OVER ()) AS Time
                FROM parquet_scan('{path}')
                WHERE pi_time BETWEEN {start_time} AND {end_time}
            """)
            
            df = con.execute("SELECT * FROM temp_data").fetchdf()
            return df
        finally:
            con.close()
    
    def get_dataset_info(self, dataset_id: str) -> dict:
        """Get information about a dataset stored in HDF5"""
        with h5py.File(self.h5_file_path, 'r') as h5f:
            if dataset_id not in h5f['processed_data']:
                raise ValueError(f"Dataset {dataset_id} not found in HDF5 file")
            
            processed_group = h5f['processed_data'][dataset_id]
            info = {
                'min_time': processed_group.attrs['min_time'],
                'max_time': processed_group.attrs['max_time'], 
                'total_samples': processed_group.attrs['total_samples'],
                'source_file': processed_group.attrs['source_file'],
            }
            return info
    
    def get_preprocessed_data(self, dataset_id: str, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Get preprocessed EEG data
        
        Returns:
        --------
        np.ndarray
            Preprocessed EEG data for the requested range
        """
        # Check if preprocessed data exists in HDF5
        with h5py.File(self.h5_file_path, 'a') as h5f:
            processed_group = h5f['processed_data']
            
            if dataset_id not in processed_group:
                # Need to compute preprocessed data
                raise RuntimeError(f"Preprocessed data for {dataset_id} not found. Please run convert_parquet_to_h5() first.")
            
            # Now load the requested slice
            dataset = processed_group[dataset_id]
            return dataset[int(start_idx):int(end_idx)]
    
    def get_windows_from_range(self, dataset_id: str, start_idx: int, end_idx: int) -> Iterator[Window]:
        """
        Generate windows from a specific range without loading entire dataset
        """
        # Get preprocessed data for the range (with some padding for windows)
        sr = self.config['sampling_rate']
        ws = self.config.get('window_size', 2.0)
        window_samples = int(ws * sr)
        
        # Add padding to ensure we can create complete windows
        padded_start = max(0, start_idx - window_samples)
        padded_end = min(self.get_dataset_info(dataset_id)['total_samples'], end_idx + window_samples)
        
        # Get the data slice
        eeg_data = self.get_preprocessed_data(dataset_id, padded_start, padded_end)
        
        # Create a temporary DataFrame for windowing
        temp_df = pd.DataFrame({
            'EEG': eeg_data,
            'pi_time': np.arange(padded_start, padded_start + len(eeg_data))  # Dummy time values
        })
        
        # Generate windows
        window_id_offset = start_idx // int((ws - self.config.get('overlap', 0.2)) * sr)
        
        for i, (win_start, win_end, segment) in enumerate(segment_windows(temp_df, self.config)):
            # Only yield windows that are within the requested range
            global_start_idx = padded_start + win_start
            if start_idx <= global_start_idx < end_idx:
                meta = WindowMeta(
                    dataset_id=dataset_id,
                    window_id=window_id_offset + i,
                    start_time=global_start_idx / sr,
                    end_time=(padded_start + win_end) / sr
                )
                yield Window(meta, segment)
    
    def __iter__(self) -> Iterator[Window]:
        """Stream windows from all datasets using HDF5 for memory-efficient access"""
        with h5py.File(self.h5_file_path, 'r') as h5f:
            for dataset_id in h5f['processed_data'].keys():
                dataset_info = self.get_dataset_info(dataset_id)
                total_samples = dataset_info['total_samples']
                
                print(f"Processing {dataset_id}: {total_samples:,} samples")
                
                # Process in chunks to maintain memory efficiency
                chunk_size_samples = self.chunk_size_minutes * 60 * self.config['sampling_rate']
                
                for start_idx in range(0, total_samples, chunk_size_samples):
                    end_idx = min(start_idx + chunk_size_samples, total_samples)
                    
                    # Generate windows from this chunk
                    for window in self.get_windows_from_range(dataset_id, start_idx, end_idx):
                        yield window
