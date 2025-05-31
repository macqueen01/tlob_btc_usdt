import hydra
from omegaconf import OmegaConf
from config.config import Config

# import os


@hydra.main(config_path="config", config_name="config")
def hydra_app(cfg: Config):
    """Main Hydra application for TLOB processing."""
    print("=== TLOB (Transformer for Limit Order Book) ===")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    print(f"Dataset Type: {cfg.dataset.type}")
    
    # Your ML pipeline logic will go here
    print("🚀 Starting TLOB processing pipeline...")


if __name__ == "__main__":
    hydra_app()
