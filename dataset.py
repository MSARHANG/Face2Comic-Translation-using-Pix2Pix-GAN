from matplotlib import pyplot as plt
from PIL import Image 
import numpy as np
import config 
import os 

from torch.utils.data import Dataset, DataLoader, random_split


class MapDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.face_dir = os.path.join(root_dir, 'face')
        self.comic_dir = os.path.join(root_dir, 'comics')
        self.list_files = sorted(os.listdir(self.face_dir))
        self.transform = config.transform
        
        
    def __len__(self):
        return len(self.list_files)
    
    
    def __getitem__(self, index):
        face_img_path = os.path.join(self.face_dir, self.list_files[index])
        comic_img_path = os.path.join(self.comic_dir, self.list_files[index])
        
        face_image = Image.open(face_img_path)
        comic_image = Image.open(comic_img_path)
        
        if self.transform:
            face_image = self.transform(face_image)
            comic_image = self.transform(comic_image)
        
        return face_image, comic_image
    
 
 
#for testing    
'''
root_dir = "dataFiles/dataset/face2comics"
dataset = MapDataset(root_dir)

val_size = 100
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) 

for face_img, comic_img in train_loader:
    show_image(face_img, comic_img, n_samples=5)
    break 
'''