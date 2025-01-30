import torch
from models.base import Model
from models.xception import XceptionModel
from models.clip_vit import CLIPViTModel
from dataset import Street2ShopImageSimilarityDataset
from dataloader import SimpleDataLoader
import torch.nn as nn
from tqdm import tqdm
import os

def calculate_validation_loss(model: Model, val_dataloader: SimpleDataLoader, criterion: nn.Module=nn.CosineEmbeddingLoss(margin=0.5), device: str="cpu"):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        batch_progress = tqdm(val_dataloader, desc="Validation", total=len(val_dataloader), leave=False)
        for street_imgs, shop_imgs, labels in batch_progress:
            
            street_imgs = street_imgs.to(device)
            shop_imgs = shop_imgs.to(device)
            labels = labels.to(device)
            
            street_embeddings = model(street_imgs)
            shop_embeddings = model(shop_imgs)
            
            loss = criterion(street_embeddings, shop_embeddings, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            cosine_sim = nn.functional.cosine_similarity(street_embeddings, shop_embeddings)
            predictions = torch.where(cosine_sim > -0.5, torch.tensor(1.0).to(device), torch.tensor(-1.0).to(device))
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            batch_progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(correct/total)*100:.2f}%'
            })
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Optionally delete variables
            del street_imgs, shop_imgs, labels, street_embeddings, shop_embeddings, loss
            
    return correct/total, total_loss / len(val_dataloader)

def train(model: Model, dataloader: SimpleDataLoader, optimizer: torch.optim.Optimizer, val_dataloader: SimpleDataLoader=None, criterion: nn.Module=nn.CosineEmbeddingLoss(margin=0.5), device: str="cpu", save_dir: str=None, max_epochs: int=3):
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(max_epochs):
        running_loss = 0.0
        batch_progress = tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{max_epochs}", total=len(dataloader), leave=False)
        
        for street_imgs, shop_imgs, labels in batch_progress:
            optimizer.zero_grad()
            
            street_imgs = street_imgs.to(device)
            shop_imgs = shop_imgs.to(device)
            labels = labels.to(device)
            
            street_embeddings = model(street_imgs)
            shop_embeddings = model(shop_imgs)
            
            loss = criterion(street_embeddings, shop_embeddings, labels)
            
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
            acc, val_loss = calculate_validation_loss(model, val_dataloader, criterion, device)
            print(f'Epoch {epoch+1}/{max_epochs} - Training loss: {avg_train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation accuracy: {acc:.4f}')
            
            # Save model if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save(os.path.join(save_dir, model.model_weights_path))
            else:
                print("Validation loss did not improve. Stopping training.")
                break
        else:
            print(f'Epoch {epoch+1}/{max_epochs} - Training loss: {avg_train_loss:.4f}')
            model.save(os.path.join(save_dir, model.model_weights_path))

if __name__ == "__main__":
    dataset = Street2ShopImageSimilarityDataset(ratio=.6)

    # Split dataset into train and validation sets (80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train_loader = SimpleDataLoader(train_dataset, batch_size=128)
    val_loader = SimpleDataLoader(val_dataset, batch_size=128)

    print(f"\n----------------------------------\n")

    # model = XceptionModel(embedding_dim=512)
    # # Adam optimizer is commonly used with XceptionNet
    # # Learning rate of 0.001 is a good starting point
    # # Weight decay (L2 regularization) helps prevent overfitting
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # model = CLIPViTModel(embedding_dim=512)
    # # AdamW optimizer is recommended for CLIP ViT models
    # # Learning rate of 5e-5 is commonly used for fine-tuning vision transformers
    # # Weight decay helps prevent overfitting while maintaining good performance
    # optimizer = torch.optim.AdamW(model.parameters(), 
    #                              lr=5e-5,
    #                              weight_decay=0.01,
    #                              betas=(0.9, 0.999))

    train(model, train_loader, optimizer, val_loader, device='cuda', save_dir='saved_models')