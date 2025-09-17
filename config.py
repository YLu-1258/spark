import os
from ruamel.yaml import YAML
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
# dataclasses for configuration sections

@dataclass
class PathsCfg:
    original_data_dir: Optional[str] = None
    data_dir: Optional[str] = 'data'
    features_dir: Optional[str] = 'features'
    segments_dir: Optional[str] = 'segments'

@dataclass
class ProjectCfg:
    sampling_rate: int
    project_name: str

@dataclass
class PreprocCfg:
    winsor: dict
    window_size : float
    overlap : float
    start_min: float
    end_min: Optional[float] = None

@dataclass
class FeaturesCfg:
    feature_types: List[str]
    wavelet: dict
    osc_bands: List[str]

@dataclass
class DetectorCfg:
    UMAP_components: int
    GMM_components: int
    UMAP_seed: int
    GMM_seed: int
    label_options: List[str]

@dataclass
class Config:
    paths: PathsCfg
    project: ProjectCfg
    preprocessing: PreprocCfg
    features: FeaturesCfg
    detection: DetectorCfg

__PathsCfg = PathsCfg(
    data_dir=None
)

__ProjectCfg = ProjectCfg(
    sampling_rate=500,
    project_name="test_project"
)

__PreprocCfg = PreprocCfg(
    start_min=0,
    end_min=None,
    winsor={"win_quant": [0.01, 0.99], "scale_quant": [0.25, 0.75], "epsilon": 1e-8},
    window_size=2.0,
    overlap=0.2
)

__FeaturesCfg = FeaturesCfg(
    feature_types=[
            "Amplitude Mean",
            "Variance",
            "Skewness",
            "Kurtosis",
            "ZCR",
            "PkPk",
            "NumPeaks",
        ],
    wavelet={"type": "db4", "level": 8},
    osc_bands=[
        "theta",
        "alpha",
        "beta",
        "gamma"
    ]
)

__DetectorCfg = DetectorCfg(
    UMAP_components=3,
    GMM_components=25,
    UMAP_seed=0,
    GMM_seed=0,
    label_options=["seizure", "normal", "artifact", "unknown"]
)

__CONFIG = Config(
    paths=__PathsCfg,
    project=__ProjectCfg,
    preprocessing=__PreprocCfg,
    features=__FeaturesCfg,
    detection=__DetectorCfg
)

def get_default_config() -> Config:
    """
    Get the default configuration.
    """
    return __CONFIG

def validate_config(config: dict) -> None:
    """
    Validate the configuration object.
    Raises ValueError if any required fields are missing or invalid.
    """
    if not config['data_dir']:
        print("Warning: paths.data_dir is not set. Manually move files into project data directory.")
    if config['sampling_rate'] <= 0:
        raise ValueError("project.sampling_rate must be a positive integer.")
    if config['start_min'] < 0:
        raise ValueError("preprocessing.start_min must be non-negative.")
    if config['window_size'] <= 0:
        raise ValueError("preprocessing.window_size must be positive.")
    if not (0 <= config['overlap'] < config['window_size']):
        raise ValueError("preprocessing.overlap must be in [0, config.preprocessing.window_size).")
    if config['UMAP_components'] <= 0:
        raise ValueError("detection.UMAP_components must be a positive integer.")
    if config['GMM_components'] <= 0:
        raise ValueError("detection.GMM_components must be a positive integer.")
    
    for feature in config['feature_types']:
        if feature not in [
            "Amplitude Mean",
            "Variance",
            "Skewness",
            "Kurtosis",
            "ZCR",
            "PkPk",
            "NumPeaks"
        ]:
            raise ValueError(f"features.feature_types contains unknown feature: {feature}")
        
    for band in config['osc_bands']:
        if band not in ["delta", "theta", "alpha", "beta", "gamma"]:
            raise ValueError(f"features.osc_bands contains unknown band: {band}")
        
    print("Configuration validated successfully.")

def update_config(project_name: str, **overrides):
    yaml = YAML()
    yaml.preserve_quotes = True
    cfg_path = os.path.join(project_name, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No config.yaml in '{project_name}'")

    # 1) Round-trip load (comments are attached to the node tree)
    with open(cfg_path) as f:
        data = yaml.load(f)

    # 2) Apply overrides in-place
    for key, val in overrides.items():
        if key not in data:
            raise KeyError(f"Unknown config key: '{key}'")
        data[key] = val

    # 3) Write back — comments and ordering are preserved
    with open(cfg_path, "w") as f:
        yaml.dump(data, f)