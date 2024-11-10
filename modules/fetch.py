from omegaconf import DictConfig
import pytorch_lightning as pl
from modules.data.data_module import DataModule
from modules.model.dnn_module import DNNModule
from modules.model.rnn_module import RNNModule


def fetch_data_module(full_config: DictConfig) -> pl.LightningDataModule:
    return DataModule(full_config)


def fetch_model_module(config: DictConfig) -> pl.LightningModule:
    model_str = config.model.type
    if model_str == 'dnn':
        return DNNModule(config)
    elif model_str == 'rnn':
        return RNNModule(config)
    
    raise NotImplementedError