import os
import tarfile
from PIL import Image
from typing import Any, Dict, Optional, Tuple
import fsspec
import hydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
import pyrootutils
from torchvision.datasets.folder import is_image_file
from tqdm import tqdm
import albumentations as A

from torchvision import utils

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from hydra import compose, initialize



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
        val_transform: Optional[A.Compose] = None,
        data_dir: str = "data/IBUG",
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_transform = train_transform
        self.val_transform = val_transform

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
            self.data_val = TransformedDlib(data_val, self.val_transform)
            self.data_test = TransformedDlib(data_test, self.val_transform)

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


    @staticmethod
    def drawBatch(batch):
        #print(f"Batch size: {batch.shape}")
        
        mean = [0.485, 0.456, 0.406]   # Assuming RGB images
        std = [0.229, 0.224, 0.225]

        image_batch, keypoints_batch = batch
        print(f"Image batch size: {image_batch.shape}")
        
        # Convert the tensor to a NumPy array and transpose it to (batch_size, height, width, channels)
        image_batch = image_batch.numpy().transpose((0, 2, 3, 1))

        print(f"Image batch size: {image_batch.shape}")

        # Denormalize the images
        image_batch = image_batch * std + mean


        
        fig, axs = plt.subplots(4, 8, figsize=(30, 10))

        for i in range(4):
            for j in range(8):
                # Display the image
                idx = (i * 8) + j 
                axs[i][j].imshow(image_batch[idx])
                axs[i][j].axis('off')

                # Extract keypoints for the current image
                keypoints = keypoints_batch[idx]

                # Rescale keypoints if necessary (e.g., if they are in a different coordinate system)
                # keypoints = rescale_keypoints(keypoints)

                # Plot keypoints on the image
                axs[i][j].scatter(keypoints[:, 0] * 224, keypoints[:, 1] * 224, s=1, c='red')

        plt.show()

        #plt.axis('off')
        plt.savefig('batchDrawers.png')







@hydra.main(version_base=None, config_path="../../configs/data", config_name='dlib.yaml')
def main(cfg : DictConfig):

    #print(OmegaConf.to_yaml(cfg))

    dataModule = hydra.utils.instantiate(cfg)
    dataModule.setup()

    train = dataModule.train_dataloader()
    batch = next(iter(train))

    DlibDataModule.drawBatch(batch)
    
    #dm = DlibDataModule(train_transform, test_transform)

    #Dlib.show_keypoints(dm.data_train[4]['image'], dm.data_train[4]['keypoints']

    #print((cfg['train_transform']._target_))


if __name__ == "__main__":
    main()