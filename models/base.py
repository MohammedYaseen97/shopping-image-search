import torch
import torch.nn as nn
from tqdm import tqdm
import os

class Model(nn.Module):
    embedding_dim: int
    criterion: nn.Module
    model_name: str
    model: nn.Module
    model_weights_path: str
    device: torch.device
    
    def __init__(self, embedding_dim):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_dim = embedding_dim
        self.criterion = nn.CosineEmbeddingLoss(margin=0.5)
    
    
    def forward(self, x):
        return self.model(x)
    
    
    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))


    def save(self, path: str):
        directory = os.path.dirname(path)
        if directory:  # Only create directory if path contains a directory part
            os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), path)


    def predict(self, dataloader):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for images in dataloader:
                embeddings.append(self.forward(images))
        return torch.cat(embeddings, dim=0)

