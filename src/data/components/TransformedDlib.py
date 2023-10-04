import albumentations as A
import hydra
from matplotlib import pyplot as plt
import numpy as np
import pyrootutils
from torch.utils.data import Dataset, DataLoader



from typing import Optional
from albumentations.pytorch import ToTensorV2

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.data.components.Dlib import Dlib




class TransformedDlib(Dataset):
    
    def __init__(self, pre_dataset : Dlib, transform: Optional[A.Compose] = None):

        self.pre_dataset = pre_dataset
        if transform: 
             self.transform = transform 
        else: 

            self.transform = A.Compose([ 
                A.Resize(224, 224), 
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                ToTensorV2()
                ], 
                    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.pre_dataset)

    def __getitem__(self, idx):
        #Actual transform the data 

        transformed = self.transform(image= np.array(self.pre_dataset[idx]['image']), keypoints=self.pre_dataset[idx]['keypoints'])
        transformed_image = transformed['image']
        transformed_keypoints = (np.array(transformed['keypoints']) / transformed_image.shape[1:]).astype(np.float32)
        # shape: (c, h , w)

        return transformed_image, transformed_keypoints
    


@hydra.main(version_base=None, config_path=".", config_name="test_transform.yaml")
def main(cfg):

    transform = hydra.utils.instantiate(cfg)

    ds = TransformedDlib(Dlib(), transform)

    print(ds[1][1].dtype)


  


if __name__ == "__main__":
    main()



