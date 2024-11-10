from omegaconf import DictConfig
from .dataset import PropheseeConcatDataset
from typing import Callable


def build_prophesee_dataset(dataset_config: DictConfig, mode: str = 'train', transform: Callable = None):
    if mode == 'train':
        mode_config = dataset_config.train
    elif mode == 'val':
        mode_config = dataset_config.val
    elif mode == 'test':
        mode_config = dataset_config.test
   
    return PropheseeConcatDataset(base_data_dir=dataset_config.data_dir,
                                  mode=mode,
                                  tau=dataset_config.tau_ms,
                                  delta_t=dataset_config.delta_t_ms,
                                  sequence_length=mode_config.sequence_length,
                                  guarantee_label=mode_config.guarantee_labels,
                                  transform=transform)