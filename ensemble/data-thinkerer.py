#Data Thinkerer
import os
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
from torchvision.transforms import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path)

            if img is not None:
                images.append(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    return images


class Modelo_Ensemble_Dataset(Dataset):
    def __init__(self, X, y, is_training=False):

        self.is_training = is_training
        
        self.base_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((720,960)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
        label = self.labels[idx]

        if self.is_training and self.aug_transform is not None:
            img = self.aug_transform(img)
        
        # Apply base transform
        img_tensor = self.base_transform(img)
        
        return img_tensor, torch.tensor(label, dtype=torch.float32)