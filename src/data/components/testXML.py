import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Dlib import Dlib
from TransformedDlib import TransformedDlib
import albumentations as A
from albumentations.pytorch import ToTensorV2

#print(box)

transform = A.Compose([
    A.Rotate(p=1),
    A.RandomBrightnessContrast(p=0.2),
    A.Resize(256, 256), 
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

A.save(transform, './transform.yaml')



#ds = Dlib()


"""
ds = TransformedDlib(Dlib(), transform)

ds.show_keypoints(0)"""

print(transform)

#txt 1008 images * 68 keypoint