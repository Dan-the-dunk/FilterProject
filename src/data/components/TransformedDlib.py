import albumentations as A
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Dlib import Dlib
from typing import Optional
from albumentations.pytorch import ToTensorV2


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
        transformed_keypoints = np.array(transformed['keypoints'])

        return {'image': transformed_image, 'keypoints': transformed_keypoints}

    def show_keypoints(self, idx):

        transformed = self.transform(image= np.array(self.pre_dataset[idx]['image']), keypoints=self.pre_dataset[idx]['keypoints'])
        transformed_image = transformed['image']
        transformed_keypoints = np.array(transformed['keypoints'])

        plt.imshow(transformed_image)
        plt.scatter(transformed_keypoints[:,0], transformed_keypoints[:,1], marker='.', c='r')
        plt.savefig('t_landmarkdrawers.png')