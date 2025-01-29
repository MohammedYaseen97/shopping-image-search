import torch
from torch.utils.data import DataLoader
from models.xception import XceptionModel
from dataset import Street2ShopImageSimilarityDataset
import random
import time
dataset = Street2ShopImageSimilarityDataset(ratio=.5)

# Split dataset into train and validation sets (80-20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

class SimpleDataLoader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_index = 0
    
    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        start_time = time.time()
        if self.current_index >= len(self.dataset):
            raise StopIteration
        
        # Get the next batch
        batch = self.dataset[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        
        # Assuming each item in the dataset is a tuple (street_img, shop_img, label)
        street_imgs, shop_imgs, labels = zip(*batch)
        
        # Convert lists of tensors to a single tensor
        street_imgs = torch.stack(street_imgs, dim=0)
        shop_imgs = torch.stack(shop_imgs, dim=0)
        labels = torch.tensor(labels)
        
        end_time = time.time()
        print(f"Time taken for batch {self.current_index // self.batch_size + 1}: {end_time - start_time:.2f} seconds")
        
        return street_imgs, shop_imgs, labels

train_loader = SimpleDataLoader(train_dataset, batch_size=32)
val_loader = SimpleDataLoader(val_dataset, batch_size=32)

print(f"\n----------------------------------\n")

# Initialize the model
model = XceptionModel(embedding_dim=512)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train(train_loader, optimizer, save_dir='saved_models')
