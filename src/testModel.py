import hydra
from omegaconf import DictConfig, OmegaConf
import pyrootutils
from lightning import LightningModule
from torchvision import models
from torch import torch

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.components.Dlib import Dlib
from src.data.components.TransformedDlib import TransformedDlib
from src.models.dlib_module import DlibModule
from src.models.components.dlib_net import DlibNet
# configs/data/train_transform/default.yaml

@hydra.main(version_base=None, config_path="../configs/data/train_transform/", config_name="default")
def main(cfg : DictConfig):

    transform = hydra.utils.instantiate(cfg)
    #print(type(transform))
    
    ds = TransformedDlib(Dlib(), transform)

    chk_path = "logs/train/runs/2023-10-04_22-34-48/checkpoints/last.ckpt"
    checkpoint = torch.load(chk_path)
    model = DlibModule.load_from_checkpoint(chk_path, net=DlibNet("resnet18", (68,2)))
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inputs, labels = inputs.to(device), labels.to(device)
    
    print(model(ds[0][0]))


def load_model_from_ckpt(path:str):
    checkpoint = torch.load(path)
    model = models.resnet18()
    model = model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    return model
  


if __name__ == "__main__":
    main()