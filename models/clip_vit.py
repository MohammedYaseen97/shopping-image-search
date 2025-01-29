import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from models.base import Model
import os

class CLIPViTModel(Model):
    def __init__(self, embedding_dim=512):
        super().__init__(embedding_dim)
        self.model_name = 'clip-vit-b-32'
        print(f'Initializing {self.model_name} on device: {self.device}..')
        
        # Load the pre-trained CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Replace the projection layer to match the desired embedding dimension
        self.clip_model.visual_projection = nn.Linear(self.clip_model.visual_projection.in_features, embedding_dim)
        self.clip_model.to(self.device)
    
    def forward(self, images):
        # Process images using the CLIP processor
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        # Get the image embeddings
        outputs = self.clip_model.get_image_features(**inputs)
        return outputs

    def save(self, path=None):
        """Save model weights to the specified path or default path"""
        directory = os.path.dirname(path)
        if directory:  # Only create directory if path contains a directory part
            os.makedirs(directory, exist_ok=True)
        torch.save(self.clip_model.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load(self, path=None):
        """Load model weights from the specified path or default path"""
        try:
            self.clip_model.load_state_dict(torch.load(path))
            print(f"Model weights loaded from {path}")
        except FileNotFoundError:
            print(f"No weights file found at {path}")
            print("Using default pretrained weights")

# Example usage:
# model = CLIPViTModel()
# model.train(train_dataloader, optimizer, criterion)
# embeddings = model.predict(test_dataloader) 