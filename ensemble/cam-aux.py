"""
Funcoes que auxiliam na geracao dos heatmaps
"""

def generate_grayscale_cam(model, target_layers, tensor_img, reshape_transform=None, size=(224, 224)):
    """
    Generates a raw grayscale Grad-CAM heatmap for a given model and image.
    
    Args:
        model (torch.nn.Module): The model to generate the CAM for.
        target_layers (list): A list of target layers in the model.
        tensor_img (torch.Tensor): Input image tensor in [0, 1] range (from ToTensor()), shape (C, H, W)
        reshape_transform (callable, optional): Function to reshape model output for CAM.
        size (tuple, optional): Target size for the input image. Defaults to (224, 224).
    Returns:
        np.ndarray: The grayscale CAM heatmap.
    """
    # Set model to eval mode
    model.eval()
    
    # Resize to target size - tensor is already in [0, 1] from ToTensor()
    img = F.interpolate(
        tensor_img.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    
    # img is already in [0, 1] range, just ensure it's float
    img = img.float()
    
    # Convert to numpy for visualization
    rgb_img_float = img.permute(1, 2, 0).cpu().numpy()
    
    # Normalize using ImageNet stats for the model input
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor = normalize(img).unsqueeze(0)  # Add batch dimension
    
    # Generate the CAM
    with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        return grayscale_cam[0, :]






def reshape_transform_vit(tensor, h=14, w=14):
    """
    Reshapes the output of a Vision Transformer.
    For ViT with 448x448 input and patch_size=32: 448/32 = 14 patches per side
    """
    # tensor shape: (batch_size, num_tokens, embedding_dim)
    # First token is [CLS], discard it
    result = tensor[:, 1:, :]
    
    # Reshape to 2D grid using provided dimensions
    result = result.reshape(tensor.size(0), h, w, tensor.size(2))
    
    # Permute to (batch_size, channels, height, width)
    result = result.permute(0, 3, 1, 2)
    return result


def reshape_transform_swin(tensor):
    """
    Reshapes the output of a Swin Transformer.
    Swin Transformers don't have a [CLS] token.
    """
    # tensor shape: (batch_size, height*width, channels)
    height = width = int(tensor.shape[1]**0.5)
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    
    # Permute to (batch_size, channels, height, width)
    result = result.permute(0, 3, 1, 2)
    return result