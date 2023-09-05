import os
import tarfile
from PIL import Image
from typing import Any, Dict, Optional, Tuple
import fsspec
import hydra
from omegaconf import DictConfig, OmegaConf
import pyrootutils
from torchvision.datasets.folder import is_image_file
from tqdm import tqdm
import albumentations as A

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from components.Dlib import Dlib
from components.TransformedDlib import TransformedDlib



class DlibDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        train_transform: Optional[A.Compose] = None,
        test_transform: Optional[A.Compose] = None,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (792, 72, 114),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        #MNIST(self.hparams.data_dir, train=True, download=True)
        #MNIST(self.hparams.data_dir, train=False, download=True)

        
        # Yup the data is here
        # Dlib is already downloaded and prepared



    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        Multiple CPU
        """
        # load and split datasets only if not loaded already
        #if not self.data_train and not self.data_val and not self.data_test:

        if not self.data_train and not self.data_val and not self.data_test:
            
            dataset = Dlib()

            data_train, data_val, data_test = random_split(
                    dataset=dataset,
                    lengths=self.hparams.train_val_test_split,
                    generator=torch.Generator().manual_seed(42),
                )

            self.data_train = TransformedDlib(data_train, self.train_transform)
            self.data_test = TransformedDlib(data_train, self.data_test)
            self.data_val = TransformedDlib(data_val)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass



@hydra.main(version_base=None, config_path="../../configs/data", config_name='dlib.yaml')
def main(cfg : DictConfig):


    #print(os.getcwd())

    print(OmegaConf.to_yaml(cfg))

    #test_transform = hydra.utils.instantiate(cfg[1])

    hydra.utils.instantiate

    #dm = DlibDataModule(train_transform, test_transform)

    #Dlib.show_keypoints(dm.data_train[4]['image'], dm.data_train[4]['keypoints']

    #print((cfg['train_transform']._target_))


if __name__ == "__main__":
    _ = main()