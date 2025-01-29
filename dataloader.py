import torch
import time

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
        batch = self.dataset[self.current_index:min(self.current_index + self.batch_size, len(self.dataset))]
        self.current_index += len(batch)
        
        # Assuming each item in the dataset is a tuple (street_img, shop_img, label)
        street_imgs, shop_imgs, labels = zip(*batch)
        
        # Convert lists of tensors to a single tensor
        street_imgs = torch.stack(street_imgs, dim=0)
        shop_imgs = torch.stack(shop_imgs, dim=0)
        labels = torch.tensor(labels)
        
        end_time = time.time()
        
        return street_imgs, shop_imgs, labels