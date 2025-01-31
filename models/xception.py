import os
import timm
import torch
import torch.nn as nn
import torchvision.models as models
from models.base import Model

class XceptionModel(Model):
    def __init__(self, embedding_dim=512, device=None):
        super().__init__(embedding_dim, device)
        self.model_name = 'xception'
        print(f'Initializing {self.model_name} on device: {self.device}..')
        
        self.model_weights_path = f'weights/{self.model_name}-{self.embedding_dim}.pt'
        
        self.model = timm.create_model('xception', pretrained=True)
        in_features = self.model.get_classifier().in_features
        self.model.fc = nn.Linear(in_features, embedding_dim)
        self.model.to(self.device)

# Example usage:
# model = XceptionModel()
# model.train(train_dataloader, optimizer, criterion)
# embeddings = model.predict(test_dataloader)
