import io
import os
import zipfile
from matplotlib.patches import Rectangle
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image;
import xml.etree.ElementTree as ET
import requests
import tqdm



class Dlib(Dataset):
    """Face keypoints dataset."""


    def __init__(self, root_dir= r'data\IBUG\ibug_300W_large_face_landmark_dataset'):
        """
        Arguments:
            xml_file (string): Path to the xml file with annotations.
            root_dir (string): Directory with all the images.
        """
        xml_path = r'data\IBUG\ibug_300W_large_face_landmark_dataset\labels_ibug_300W.xml'
        #Download the damn file from the net bro
        if(not ET.parse(xml_path)):
            #download
            download_data()
            unzip_data()
        else:
        #
            tree = ET.parse(xml_path)
            self.root = tree.getroot()

            #change dis code
            self.root_dir = root_dir

    def __len__(self):
        return len(self.root[2])
    

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
                                self.root[2][idx].attrib['file'])
        image = Image.open(img_name).convert("RGB")

        keypoints = []
        for kp in self.root[2][idx][0].iter():
            keypoints.append([kp.attrib.get('x'), kp.attrib.get('y')])

        keypoints = keypoints[1:]
        keypoints = np.array(keypoints, dtype=float)

        #keypoints = np.array([keypoints], dtype=float).reshape(-1, 2)
        

        #Crop the image here.
        #Get box (top_left, width, height).
        
        box_dict = self.root[2][idx][0].attrib

        box = [box_dict.get('left'), box_dict.get('top'), 
                float(box_dict.get('left'))+float(box_dict.get('width')), 
                float(box_dict.get('top')) + float(box_dict.get('height'))]
        box = np.array(box, dtype=float)

        image = image.crop(box=box)
        # Crop the landmark

        keypoints[:,0] = keypoints[:,0] - float(box_dict.get('left'))
        keypoints[:,1] = keypoints[:,1] - float(box_dict.get('top')) 

        sample = {'image': image, 'keypoints': keypoints}
        return sample


    @staticmethod
    def show_keypoints(image, keypoints):
        plt.imshow((image.permute(1,2,0)))
        plt.scatter(keypoints[:,0] * 224, keypoints[:,1] * 224, marker='.', c='r')
        plt.savefig('landmarkdrawers.png')

    @staticmethod
    def show_boundingboxes(image, keypoints, box):

        plt.scatter(keypoints[:,0], keypoints[:,1], marker='.', c='r')

        box_width, box_height = 244

        # Get the current reference
        ax = plt.gca()

        # Create a Rectangle patch
        rect = Rectangle( (box[0],box[1] )
                            , box_width, box_height ,linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.imshow(denormalize(image))
        plt.savefig('bbdrawers.png')

    @staticmethod
    def testImage(image):
        t = image.clone().permute(1, 2, 0)
        print(f"Max: {torch.max(t)}, Min: {torch.min(t)}")



IMG_MEAN = [0.485, 0.456, 0.406] 
IMG_STD = [0.229, 0.224, 0.225]
def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch. Tensor:
# IN: 3, H, W
    ten = x.clone().permute(1, 2, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    

    print(f"Max: {torch.max(t)}, Min: {torch.min(t)}")
    # B, 3, H, W
    return ten 






def download_data():

    url = "http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz"
    response = requests.get(url, stream=True)
    with open(r"data/FilerData.zip", mode="wb") as file:
        dl = 0
        for chunk in tqdm(response.iter_content(chunk_size=1024)): 
            if chunk:
                file.write(chunk)
                file.flush()
                    
        
def unzip_data():

    with zipfile.ZipFile('data\FilerData.zip', 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc='Extracting '):
            try:
                zip_ref.extract(member, 'data')
            except zipfile.error as e:
                pass



