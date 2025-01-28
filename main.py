import torch
from torch.utils.data import DataLoader
from models.xception import XceptionModel
from dataset import Street2ShopImageSimilarityDataset

dataset = Street2ShopImageSimilarityDataset(ratio=.5)

# Split dataset into train and validation sets (80-20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# Use DataLoader with the custom collate function
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"\n----------------------------------\n")

# Initialize the model
model = XceptionModel(embedding_dim=512)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train(train_loader, val_loader, optimizer, save_dir='saved_models')
