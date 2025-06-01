import hydra
from omegaconf import OmegaConf
from config.config import Config
from constants import DatasetType
from preprocessing.btc import BTCDataBuilder

# import os


@hydra.main(config_path="config", config_name="config")
def hydra_app(cfg: Config):
    if cfg.dataset.type != DatasetType.BTC_USDT_SPOT:
        raise ValueError(f"Dataset type {cfg.dataset.type} is not supported")
    
    btc_data_builder = BTCDataBuilder(
        data_dir=cfg.dataset.data_dir,
        sampling_time=cfg.dataset.sampling_time,
        dates=cfg.dataset.dates,
        split_rates=cfg.dataset.split_rates,
        split_days=cfg.dataset.split_days
    )
    btc_data_builder.prepare_and_save()

if __name__ == "__main__":
    hydra_app()
