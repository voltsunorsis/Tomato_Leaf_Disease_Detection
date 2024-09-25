import torch
import torch.nn as nn
from torchvision import models

def create_model(num_classes):
    # Load the pre-trained EfficientNet-B0 model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze all the parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the last fully connected layer
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model

# If you need to get the number of classes dynamically (optional)
def get_num_classes(model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    num_classes = state_dict['classifier.1.weight'].size(0)
    return num_classes