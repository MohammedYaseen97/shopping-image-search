import os
import timm
import torch
import torch.nn as nn
import torchvision.models as models
from models.base import Model

class XceptionModel(Model):
    def __init__(self, embedding_dim=512):
        super().__init__(embedding_dim)
        self.model_name = 'xception'
        print(f'Initializing {self.model_name} on device: {self.device}..')
        
        self.model_weights_path = 'xception.pth'
        
        self.model = timm.create_model('xception', pretrained=True)
        in_features = self.model.get_classifier().in_features
        self.model.fc = nn.Linear(in_features, embedding_dim)
        self.model.to(self.device)
    
    def save(self, path=None):
        """Save model weights to the specified path or default path"""
        directory = os.path.dirname(path)
        if directory:  # Only create directory if path contains a directory part
            os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load(self, path=None):
        """Load model weights from the specified path or default path"""
        try:
            self.model.load_state_dict(torch.load(path))
            print(f"Model weights loaded from {path}")
        except FileNotFoundError:
            print(f"No weights file found at {path}")
            print("Using default pretrained weights")

# Example usage:
# model = XceptionModel()
# model.train(train_dataloader, optimizer, criterion)
# embeddings = model.predict(test_dataloader)
