

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_images_from_folder(folder_path, H, W):
    """Load all images as preprocessed tensors"""
    images = []
    for filename in os.listdir(folder_path):
        print("1 ", end='')
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('RGB')
        # Apply transforms immediately
        transform = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               #std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img)
        images.append(img_tensor)
    return images

class Model_Dataset(Dataset):
    """
    @param X: Lista de PIL images
    @param y: Labels
    """
    def __init__(self, images, labels, is_training=False):
        
        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(0.5)
        ]) if is_training else None

                
        self.images = images  # Already tensors
        self.labels = labels
        self.is_training = is_training
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        
        if self.is_training:
            img = self.aug_transform(img)
        
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)