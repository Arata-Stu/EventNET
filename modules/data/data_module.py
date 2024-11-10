from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig

from data.dataset.build_dataset import build_dataset
from data.data_utils.collate import custom_collate_fn

class DataModule(pl.LightningDataModule):

    def __init__(self, full_config: DictConfig):
        super().__init__()
        self.full_config = full_config
        self.dataset_config = full_config.dataset

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = build_dataset(dataset_config=self.dataset_config, mode='train')
            self.valid_dataset = build_dataset(dataset_config=self.dataset_config, mode='val')
    
            print(f"train dataset size: {len(self.train_dataset)}")
            print(f"valid dataset size: {len(self.valid_dataset)}")

        if stage == 'test':
            self.test_dataset = build_dataset(dataset_config=self.dataset_config, mode='test')
            print(f"test dataset size: {len(self.test_dataset)}")


    def train_dataloader(self):
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.full_config.experiment.dataset.train.batch_size,
            shuffle=True,
            num_workers=self.full_config.experiment.dataset.train.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )
        
    def val_dataloader(self):
        
        return DataLoader(
            self.valid_dataset,
            batch_size=self.full_config.experiment.dataset.val.batch_size,
            shuffle=False,
            num_workers=self.full_config.experiment.dataset.val.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
        
    def test_dataloader(self):
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.full_config.experiment.dataset.test.batch_size,
            shuffle=False,
            num_workers=self.full_config.experiment.dataset.test.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
    
        