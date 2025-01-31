import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from models.base import Model
import os

class CLIPViTModel(Model):
    def __init__(self, embedding_dim=512, device=None):
        super().__init__(embedding_dim, device)
        self.model_name = 'clip-vit-b-32'
        print(f'Initializing {self.model_name} on device: {self.device}..')
        
        # Load just the vision model from CLIP
        self.vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model
        self.normalization = True
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.model_weights_path = f'weights/{self.model_name}-{self.embedding_dim}.pt'
        
        # Replace the projection layer to match the desired embedding dimension
        self.projection = nn.Linear(self.vision_model.config.hidden_size, embedding_dim)
        
        # Move models to device
        self.vision_model.to(self.device)
        self.projection.to(self.device)
    
    def forward(self, images):
        outputs = self.vision_model(pixel_values=images)
        embeddings = self.projection(outputs.pooler_output)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True) # normalize embeddings
        return embeddings

    def save(self, path=None):
        """Save model weights to the specified path or default path"""
        if path is None:
            path = self.model_weights_path
        
        directory = os.path.dirname(path)
        if directory:  # Only create directory if path contains a directory part
            os.makedirs(directory, exist_ok=True)
            
        # Save both vision model and projection layer weights
        state_dict = {
            'vision_model': self.vision_model.state_dict(),
            'projection': self.projection.state_dict()
        }
        try:
            torch.save(state_dict, path)
            print(f"Model weights saved to {path}")
        except Exception as e:
            print(f"Failed to save model weights: {e}")

    def load(self, path=None):
        """Load model weights from the specified path or default path"""
        if path is None:
            path = self.model_weights_path
            
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.vision_model.load_state_dict(state_dict['vision_model'])
            self.projection.load_state_dict(state_dict['projection'])
            print(f"Model weights loaded from {path}")
        except FileNotFoundError:
            print(f"No weights file found at {path}")
            print("Using default pretrained weights")
        except KeyError as e:
            print(f"Key error during loading model weights: {e}")
        except Exception as e:
            print(f"Failed to load model weights: {e}")

# Example usage:
# model = CLIPViTModel()
# model.train(train_dataloader, optimizer, criterion)
# embeddings = model.predict(test_dataloader) 