import torch  # Import the PyTorch library
import torchvision
import torch.nn as nn

# Define the model architecture (MobileNetV2)
mobilenetv2 = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')  # Load pre-trained weights
# Adjust the final classification layer
num_ftrs = mobilenetv2.classifier[1].in_features  # Get the number of input features for the last layer
mobilenetv2.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),  # First linear layer with 512 units
    nn.GELU(),  # GELU activation function
    nn.Linear(512, 32),  # Second linear layer with 32 units
    nn.GELU(),  # GELU activation function
    nn.Linear(32, 1)   # Output layer with 1 unit (for age prediction)
)