"""
Funcoes que auxiliam na geracao dos heatmaps
"""

def generate_grayscale_cam(model, target_layers, pil_img, reshape_transform=None, size=(224, 224)):
    """
    Generates a raw grayscale Grad-CAM heatmap for a given model and image.
    This function now supports both CNNs and Transformers via the `reshape_transform` parameter.
    
    Args:
        model (torch.nn.Module): The model to generate the CAM for.
        target_layers (list): A list of target layers in the model.
        image_path (str): The path to the input image.
        reshape_transform (callable, optional): A function to reshape the model's output for CAM.
                                                 Required for Transformer models. Defaults to None.
        size (tuple, optional): The target size for the input image. Defaults to (224, 224).

    Returns:
        np.ndarray: The grayscale CAM heatmap.
    """
    
    # Preprocess the image to a consistent size
    rgb_img_float = np.array(pil_img.resize(size), dtype=np.float32) / 255.0
    input_tensor = preprocess_image(rgb_img_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Generate the CAM
    with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        # Get the first (and only) image from the batch
        return grayscale_cam[0, :]





def reshape_transform_vit(tensor):
    """
    Reshapes the output of a Vision Transformer.
    Removes the [CLS] token and reshapes the remaining tokens into a 2D feature map.
    """
    # tensor shape: (batch_size, num_tokens, embedding_dim)
    # The first token is the [CLS] token, we discard it.
    result = tensor[:, 1:, :]
    
    # Reshape to a 2D grid
    height = width = int(result.shape[1]**0.5)
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    
    # Permute to (batch_size, channels, height, width)
    result = result.permute(0, 3, 1, 2)
    return result

def reshape_transform_swin(tensor):
    """
    Reshapes the output of a Swin Transformer.
    Swin Transformers do not have a [CLS] token, so we just reshape the output.
    """
    # tensor shape: (batch_size, height*width, channels)
    height = width = int(tensor.shape[1]**0.5)
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    
    # Permute to (batch_size, channels, height, width)
    result = result.permute(0, 3, 1, 2)
    return result