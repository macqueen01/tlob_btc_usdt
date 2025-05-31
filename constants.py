from enum import Enum

class DatasetType(Enum):
    BTC_USDT_SPOT = "BTC_USDT_SPOT"

class AcceleratorType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    # TPU = "tpu"
    MPS = "mps"

class SamplingTime(Enum):
    TWO_HUNDRED_MS = "200ms"
    FIVE_HUNDRED_MS = "500ms"
    ONE_SECOND = "1s"
    TWO_SECONDS = "2s"
    FIVE_SECONDS = "5s"
    TEN_SECONDS = "10s"
    
RAW_DATA_DIR = "./preprocessed"
N_LOB_LEVELS = 10
LEN_SMOOTH = 10
HORIZONS = [10, 20, 50, 100]
    