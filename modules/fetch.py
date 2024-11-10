from omegaconf import DictConfig
import pytorch_lightning as pl
from modules.data.data_module import DataModule

def fetch_data_module(full_config: DictConfig) -> pl.LightningDataModule:
    return DataModule(full_config)