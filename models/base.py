import torch
import torch.nn as nn
from tqdm import tqdm
import os

class Model(nn.Module):
    embedding_dim: int
    criterion: nn.Module
    model_name: str
    model: nn.Module
    normalization: bool = False
    model_weights_path: str
    device: torch.device
    
    def __init__(self, embedding_dim, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.criterion = nn.CosineEmbeddingLoss(margin=0.5)
    
    
    def forward(self, x):
        return self.model(x)
    

    def load(self, path=None):
        """Load model weights from the specified path or default path"""
        if path is None:
            path = self.model_weights_path
        
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model weights loaded from {path}")
        except FileNotFoundError:
            print(f"No weights file found at {path}")
            print("Using default pretrained weights")
        except Exception as e:
            print(f"Failed to load model weights: {e}")
    
    
    def save(self, path=None):
        """Save model weights to the specified path or default path"""
        if path is None:
            path = self.model_weights_path
        
        directory = os.path.dirname(path)
        if directory:  # Only create directory if path contains a directory part
            os.makedirs(directory, exist_ok=True)
        
        try:
            torch.save(self.model.state_dict(), path)
            print(f"Model weights saved to {path}")
        except Exception as e:
            print(f"Failed to save model weights: {e}")

