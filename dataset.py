import os
import numpy as np
import pandas as pd
from .preprocessing.apply_preprocessing import apply_preprocessing
from .preprocessing.segment_windows import segment_windows
from dataclasses import dataclass
from pathlib import Path

@dataclass
class WindowMeta:
    dataset_id: str         # e.g. the raw filename or a UUID
    window_id: int          # sequential index
    start_time: float       # seconds into the file
    end_time: float

@dataclass
class Window:
    meta: WindowMeta
    data: np.ndarray 

class EEGDataset:
    def __init__(self, config):
        self.raw_files = self.list_raw_files(config['data_dir'])
        self.config  = config

    def list_raw_files(self, directory_path, raw_extensions=['.parquet', '.csv', '.xlxs']):
        raw_files = []
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in raw_extensions):
                raw_files.append(os.path.join(directory_path, filename))
        return raw_files
    
    def load_parquet(self, path):
        df = pd.read_parquet(path)
        return df
    
    def load_excel(self, path):
        df = pd.read_excel(path)
        return df
    
    def load_csv(self, path):
        df = pd.read_csv(path)
        return df
    
    def get_num_samples(self, path):
        if path.endswith('.parquet'):
            df = self.load_parquet(path)
        elif path.endswith('.xlxs'):
            df = self.load_excel(path)
        elif path.endswith('.csv'):
            df = self.load_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        return len(df)
    
    def __iter__(self):
        for path in self.raw_files:
            dataset_id = Path(path).stem
            if (path.endswith('.parquet')):
                df = self.load_parquet(path)
            if (path.endswith('.xlxs')):
                df = self.load_excel(path)
            elif (path.endswith('.csv')):
                df = self.load_csv(path)
            preprocessed_eeg = apply_preprocessing(df, self.config)

            # Add Time column if not present
            if 'Time' not in df.columns:
                df['Time'] = df['pi_time'] - df['pi_time'].min()
            df['lfp_hpf_5'] = preprocessed_eeg

            for i, (start, end, segment) in enumerate(segment_windows(df, self.config)):
                meta = WindowMeta(dataset_id, i, start, end)
                yield Window(meta, segment)