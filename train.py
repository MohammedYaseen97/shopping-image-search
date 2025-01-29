import torch
from models.xception import XceptionModel
from dataset import Street2ShopImageSimilarityDataset
from dataloader import SimpleDataLoader

dataset = Street2ShopImageSimilarityDataset(ratio=.5)

# Split dataset into train and validation sets (80-20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

train_loader = SimpleDataLoader(train_dataset, batch_size=128)
val_loader = SimpleDataLoader(val_dataset, batch_size=128)

print(f"\n----------------------------------\n")

model = XceptionModel(embedding_dim=512)
# Adam optimizer is commonly used with XceptionNet
# Learning rate of 0.001 is a good starting point
# Weight decay (L2 regularization) helps prevent overfitting
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


model.train(train_loader, optimizer, val_loader, save_dir='saved_models')