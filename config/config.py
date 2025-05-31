from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field

from omegaconf import MISSING
from constants import DatasetType, AcceleratorType, SamplingTime
    
@dataclass
class Dataset:
    type: DatasetType = MISSING
    data_dir: str = MISSING
    batch_size: int = 32
    dates: list = MISSING
    sampling_time: SamplingTime = MISSING
    
@dataclass
class BTC_USDT_SPOT(Dataset):
    type: DatasetType = DatasetType.BTC_USDT_SPOT
    data_dir: str = "./preprocessed"
    dates: list = field(default_factory=lambda: ["2023-06-17", "2025-05-20"])
    batch_size: int = 128
    sampling_time: SamplingTime = SamplingTime.TWO_HUNDRED_MS
    
@dataclass
class Config:
    dataset: Dataset = MISSING
    accelerator: AcceleratorType = AcceleratorType.MPS
    
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="dataset", name="btc_usdt_spot", node=BTC_USDT_SPOT)