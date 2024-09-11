from PIL import Image 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision 
import torchvision.models as models 
import torchvision.transforms as transforms 

class SquarePad:
    """Pads image to make it square.

    Pads a PIL Image or a PyTorch tensor with zeros to make it square,
    keeping the aspect ratio and centering the original image.

    Args:
        image: A PIL Image or a PyTorch tensor representing the image.

    Returns:
        A PyTorch tensor representing the padded image.

    Raises:
        TypeError: If the input image is not a PIL Image or a PyTorch tensor.
    """
    def __call__(self, image):
        # Check if the input is a PIL Image or a PyTorch tensor.
        if not isinstance(image, (Image.Image, torch.Tensor)):
            # If not, raise a TypeError.
            raise TypeError("Input image must be a PIL Image or a PyTorch tensor.")
        # Get the shape of the image.
        s = image.shape
        # Calculate the maximum width and height.
        max_wh = max(s[-1], s[-2])
        # Calculate the horizontal padding.
        hp = int((max_wh - s[-1]) / 2)
        # Calculate the vertical padding.
        vp = int((max_wh - s[-2]) / 2)
        # Create the padding tuple.
        padding = (hp, hp, vp, vp)
        # Pad the image with zeros and return the result.
        return F.pad(image, padding, 'constant', 0)


class norm_to_zo:
    """Normalizes image to the range [0, 1].

    Normalizes a PIL Image or a PyTorch tensor to the range [0, 1]
    by dividing all pixel values by 255.

    Args:
        image: A PIL Image or a PyTorch tensor representing the image.

    Returns:
        A PyTorch tensor representing the normalized image.

    Raises:
        TypeError: If the input image is not a PIL Image or a PyTorch tensor.
    """
    def __call__(self, image):
        # Check if the input is a PIL Image or a PyTorch tensor.
        if not isinstance(image, (Image.Image, torch.Tensor)):
            # If not, raise a TypeError.
            raise TypeError("Input image must be a PIL Image or a PyTorch tensor.")
        # Divide all pixel values by 255 and return the result.
        return image/255.0

class FeatureExtractorNet:
    """Extracts features from an image using a pretrained convolutional neural network.

    Initializes a feature extractor with a specified model and device,
    loads the pretrained model, and extracts features from an input image.

    Args:
        model_name: The name of the pretrained model to use (default: 'resnet18').
        device: The device to run the model on (default: 'cpu').

    Attributes:
        model_name: The name of the pretrained model.
        device: The device to run the model on.
        model: The pretrained model used for feature extraction.
    """
    def __init__(self, model_name='resnet18', device=torch.device('cpu')):
        # Initialize the model name and device.
        self.model_name = model_name
        self.device = device
        # Load the pretrained model.
        self.model = self.get_model()

    def get_model(self):
        """Loads the pretrained model based on the model name.

        Loads a pretrained convolutional neural network (ResNet18 or RegNetY)
        and sets its parameters to not require gradients.

        Returns:
            A PyTorch Sequential model for feature extraction.
        """
        # Check if the model name is 'resnet18'.
        if self.model_name == 'resnet18':
            # If so, load the pretrained ResNet18 model.
            model_conv = models.resnet18(pretrained=True)

        # Check if the model name is 'regnet_y'.
        if self.model_name == 'regnet_y':
            # If so, load the pretrained RegNetY model.
            model_conv = models.regnet_y_16gf(pretrained = True)
        # Add more models as needed, e.g., elif self.model_name == 'resnet50': ...

        # Freeze the model parameters.
        for param in model_conv.parameters():
            param.requires_grad = False

        # Create a Sequential model for feature extraction. Deleting last layer.
        model_feature = torch.nn.Sequential(*(list(model_conv.children())[:-1]))
        # Return the model on the specified device.
        return model_feature.to(self.device)


    def extract_features(self, img):
        """Extracts features from an image.

        Sets the model to evaluation mode, applies transformations to the image,
        and passes it through the model to extract features.

        Args:
            img: A PIL Image representing the image.

        Returns:
            A PyTorch tensor representing the extracted features.
        """
        # Set the model to evaluation mode.
        self.model.eval()
        # Define the transformations to apply to the image.
        transform = transforms.Compose([
            # Convert the image to a PyTorch tensor.
            transforms.ToTensor(),
            # Pad the image to make it square.
            SquarePad(),
            # Resize the image to 224x224 pixels.
            transforms.Resize(224),
            # Center crop the image to 224x224 pixels.
            transforms.CenterCrop(224),
            # Normalize the image.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Apply the transformations to the image.
        img = transform(img)
        # Add a batch dimension to the image.
        img = img.unsqueeze(0)
        # Move the image to the specified device.
        img = img.to(self.device)
        # Extract features from the image.
        with torch.no_grad():
            features = self.model(img)
        # Return the extracted features.
        return features
