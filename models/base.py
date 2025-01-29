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

    def calculate_validation_loss(self, val_dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            batch_progress = tqdm(val_dataloader, desc="Validation", total=len(val_dataloader), leave=False)
            for street_imgs, shop_imgs, labels in batch_progress:
                
                street_imgs = street_imgs.to(self.device)
                shop_imgs = shop_imgs.to(self.device)
                labels = labels.to(self.device)
                
                street_embeddings = self.forward(street_imgs)
                shop_embeddings = self.forward(shop_imgs)
                
                loss = self.criterion(street_embeddings, shop_embeddings, labels)
                total_loss += loss.item()
                
                batch_progress.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Optionally delete variables
                del street_imgs, shop_imgs, labels, street_embeddings, shop_embeddings, loss
                
        return total_loss / len(val_dataloader)

    def train(self, dataloader, optimizer, val_dataloader=None, save_dir=None, max_epochs=3):
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(max_epochs):
            running_loss = 0.0
            batch_progress = tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{max_epochs}", total=len(dataloader), leave=False)
            
            for street_imgs, shop_imgs, labels in batch_progress:
                optimizer.zero_grad()
                
                street_imgs = street_imgs.to(self.device)
                shop_imgs = shop_imgs.to(self.device)
                labels = labels.to(self.device)
                
                street_embeddings = self.forward(street_imgs)
                shop_embeddings = self.forward(shop_imgs)
                
                loss = self.criterion(street_embeddings, shop_embeddings, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                batch_progress.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Optionally delete variables
                del street_imgs, shop_imgs, labels, street_embeddings, shop_embeddings, loss
            
            avg_train_loss = running_loss / len(dataloader)
            
            if val_dataloader:
                val_loss = self.calculate_validation_loss(val_dataloader)
                print(f'Epoch {epoch+1}/{max_epochs} - Training loss: {avg_train_loss:.4f}, Validation loss: {val_loss:.4f}')
                
                # Save model if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save(os.path.join(save_dir, self.model_weights_path))
                else:
                    print("Validation loss did not improve. Stopping training.")
                    break
            else:
                print(f'Epoch {epoch+1}/{max_epochs} - Training loss: {avg_train_loss:.4f}')
                self.save(os.path.join(save_dir, self.model_weights_path))
            

    def predict(self, dataloader):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for images in dataloader:
                embeddings.append(self.forward(images))
        return torch.cat(embeddings, dim=0)

