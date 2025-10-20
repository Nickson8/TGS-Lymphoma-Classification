

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

class Model_Dataset(Dataset):
    """
    @param X: Lista de PIL images
    @param y: Labels
    """
    def __init__(self, X, y, input_size, is_training: bool = False):
        
        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(0.5)
        ]) if is_training else None
        
        self.base_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((input_size[0], input_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

                
        self.labels = y
        
        self.all_images = X

        self.is_training = is_training
    
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