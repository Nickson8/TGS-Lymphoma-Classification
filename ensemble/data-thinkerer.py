#Data Thinkerer
import os
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
from torchvision.transforms import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_images_from_folder(folder_path):
    images = []
    nomes = []
    
    for filename in os.listdir(folder_path):
        print("1 ", end='')
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('RGB')
        # Apply transforms immediately
        transform = transforms.Compose([
            transforms.Resize((720,960)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img)
        images.append(img_tensor)
        nomes.append(filename.split('/')[-1].split('.')[0])
        
    return images, nomes


class Modelo_Ensemble_Dataset(Dataset):
    def __init__(self, X, y, is_training=False):

        self.is_training = is_training

        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(0.5)
        ]) if is_training else None

        self.labels = y        
        self.all_images = X

    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img = self.all_images[idx]
        
        if self.is_training:
            img = self.aug_transform(img)
        
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)