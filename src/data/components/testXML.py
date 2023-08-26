import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Dlib import Dlib
from TransformedDlib import TransformedDlib
import albumentations as A

#print(box)

transform = A.Compose([
    A.Rotate(p=1),
    A.RandomBrightnessContrast(p=0.2),
    A.Resize(224, 224), 
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ToTensorV2()
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

#ds = Dlib()

ds = TransformedDlib(Dlib(), transform)

"""plt.imshow(ds[40]['image'])
plt.scatter(ds[40]['keypoints'][:,0], ds[40]['keypoints'][:,1], marker='.', c='r')

plt.savefig('bood.png')"""

"""ds = Dlib()
ds.show_keypoints(40)"""

ds.show_keypoints(0)


#txt 1008 images * 68 keypoint