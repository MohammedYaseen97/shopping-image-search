from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset
import random
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset
import torch

SEED = 42

# Shop images are clean product photos, street images are user photos with crops
shop_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),  # Product may be flipped in street photo
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Slight color variation
])

street_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # More color variation
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Perspective variation
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # Blur from camera motion/quality
])

# Define a separate ToTensor and Normalize transform
to_tensor_and_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def identify_valid_rows(dataset_imgs):
    valid_idx = []

    for i in tqdm(range(len(dataset_imgs)), desc="Identifying valid images"):
        try:
            item = dataset_imgs[i]            
            valid_idx.append(i)     # If no exception is raised, add the index to valid_idx
        except Exception as e:
            print(f"Error processing image at index {i}: {e}")
            continue  # Skip the corrupted image

    return valid_idx

def get_cropped_image(item):
    street_img = item['street_photo_image']
    left = int(item['left'])
    top = int(item['top'])
    width = int(item['width'])
    height = int(item['height'])
    return street_img.crop((left, top, left + width, top + height))

def transform_images(example):
    try:
        if example['street_photo_image'].mode != 'RGB':
            example['street_photo_image'] = example['street_photo_image'].convert('RGB')
        example['street_photo_image'] = get_cropped_image(example)
        example['street_photo_image'] = street_transform(example['street_photo_image'])
        
        if example['shop_photo_image'].mode != 'RGB':
            example['shop_photo_image'] = example['shop_photo_image'].convert('RGB')
        example['shop_photo_image'] = shop_transform(example['shop_photo_image'])
        
        # Mark the example as valid
        return {**example, 'valid': True}
    except Exception as e:
        print(f"Error processing image: {e}")
        # Mark the example as invalid
        return {**example, 'valid': False}

class Street2ShopImageSimilarityDataset(Dataset):
    def __init__(self, shop_transform=shop_transform, street_transform=street_transform, num_negative_pairs=2, ratio=1.0):
        print('Initializing Street2ShopImageSimilarityDataset..')
        
        self.shop_transform = shop_transform
        self.street_transform = street_transform
        dataset_path = f'street2shop_{ratio}' if ratio < 1 else 'street2shop'
        dataset_path_without_ratio = dataset_path.rstrip(f'_{ratio}')
        dataset_imgs_path = f'street2shop_imgs_{ratio}' if ratio < 1 else 'street2shop_imgs'
        self.num_negative_pairs = num_negative_pairs
        sampled_indices = None
        
        # Load the dataset from disk, remove the image columns to save memory
        try:
            print(f"Loading dataset from {dataset_path}...")
            
            self.dataset = load_from_disk(dataset_path)
            
            # in case we load the original street2shop dataset
            if 'index' not in self.dataset.column_names:
                self.dataset = self.dataset['train'].remove_columns(['type', 'street_photo_image', 'shop_photo_image'])
                self.dataset = self.dataset.add_column('index', list(range(len(self.dataset))))
        except FileNotFoundError:
            print(f"Dataset not found at {dataset_path}. Loading {dataset_path_without_ratio} dataset...")
            
            try:
                self.dataset = load_from_disk(dataset_path_without_ratio)['train'].remove_columns(['type', 'street_photo_image', 'shop_photo_image'])
            except FileNotFoundError:
                self.dataset = load_dataset(dataset_path_without_ratio)['train'].remove_columns(['type', 'street_photo_image', 'shop_photo_image'])
        
            # Add an index column to the dataset
            self.dataset = self.dataset.add_column('index', list(range(len(self.dataset))))
            self.sample_dataset(ratio)
            
            self.dataset.save_to_disk(dataset_path)
        
        print("Length of dataset:", len(self.dataset))
        sampled_indices = list(self.dataset['index'])
        
        # Group images by category for efficient negative sampling
        self.category_groups = {}
        for idx, item in enumerate(self.dataset):
            category = item['category']
            if category not in self.category_groups:
                self.category_groups[category] = []
            self.category_groups[category].append(idx)
        
        # Precompute pairs
        self.pairs = self._generate_pairs()
        print("Sample pair structure:", self.pairs[0])  # Debugging line
        print('# of pairs:', len(self.pairs))
        del self.dataset # Free up memory
        
        # Load dataset images
        valid_idx = []
        try:
            print(f"Loading dataset images from {dataset_imgs_path}...")
            
            self.dataset_imgs = load_from_disk(dataset_imgs_path)
            
            valid_idx = list(self.dataset_imgs['index'])
        except FileNotFoundError:
            print(f"Dataset images not found at {dataset_imgs_path}. Loading {dataset_path_without_ratio} dataset images and performing transforms...")
            
            try:
                self.dataset_imgs = load_from_disk(dataset_path_without_ratio)['train'].select_columns(['street_photo_image', 'shop_photo_image', 'left', 'top', 'width', 'height'])
            except FileNotFoundError:
                self.dataset_imgs = load_dataset(dataset_path_without_ratio)['train'].select_columns(['street_photo_image', 'shop_photo_image', 'left', 'top', 'width', 'height'])
            self.dataset_imgs = self.dataset_imgs.select(sampled_indices)
            
            # adding new index column to track the index of the uncorrupted images
            self.dataset_imgs = self.dataset_imgs.add_column('index', list(range(len(self.dataset_imgs))))
            
            # Extract rows with uncorrupted images
            valid_idx = identify_valid_rows(self.dataset_imgs)

            # Filter the dataset to only include rows with uncorrupted images
            self.dataset_imgs = self.dataset_imgs.select(valid_idx)
            
            # Apply transformations with tqdm
            self.dataset_imgs = self.dataset_imgs.map(
                lambda example: transform_images(example),
                desc="Transforming images"
            )
            
            # Rows with further issues while transforming .. to be subtracted from valid_idx
            invalid_idx_set = set([i for i in range(len(self.dataset_imgs)) if not self.dataset_imgs[i]['valid']])
            
            # Filter valid_idx to exclude newly discovered invalid indices
            valid_idx = [i for i in valid_idx if i not in invalid_idx_set]
            
            self.dataset_imgs.save_to_disk(dataset_imgs_path)
        
        valid_idx_set = set(valid_idx)
        print('# of valid idx set:', len(valid_idx_set))
        
        # Filter pairs to only include valid indices
        self.pairs = [(s_idx, sh_idx, label) for s_idx, sh_idx, label in self.pairs if s_idx in valid_idx_set and sh_idx in valid_idx_set]
        print('# of pairs after filtering:', len(self.pairs))
        
    def sample_dataset(self, ratio):
        if ratio == 1:
            return
        
        # Convert the dataset to a Pandas DataFrame
        df = self.dataset.to_pandas()
        
        # Perform stratified split
        train_df, _ = train_test_split(
            df,
            test_size=ratio,
            stratify=df['category'],
            random_state=SEED
        )
        
        # Convert back to a Hugging Face dataset
        self.dataset = Dataset.from_pandas(train_df)
            
    def _generate_pairs(self):
        pairs = []
        for idx, item in tqdm(enumerate(self.dataset), desc="Generating pairs", total=len(self.dataset)):
            # Positive pair
            pairs.append((idx, idx, 1))
            
            # Generate negative pairs
            neg_indices = set()
            while len(neg_indices) < self.num_negative_pairs:
                neg_idx = self._get_negative_pair(idx, item)
                if neg_idx not in neg_indices:  # Ensure unique negative pairs
                    neg_indices.add(neg_idx)
                    pairs.append((idx, neg_idx, -1))
        
        return pairs

    def _is_different_item(self, item1, item2):
        """Check if two items are different based on image and crop coordinates"""
        return (item1['street_photo_url'] != item2['street_photo_url'] or
                item1['left'] != item2['left'] or
                item1['top'] != item2['top'] or
                item1['width'] != item2['width'] or
                item1['height'] != item2['height'])

    def _get_negative_pair(self, idx, item):
        """Generate a negative pair for a given item"""
        category_indices = self.category_groups[item['category']]
        while True:
            neg_idx = random.choice(category_indices)
            neg_item = self.dataset[neg_idx]
            if self._is_different_item(item, neg_item):
                return neg_idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        def _process_single_item(idx):
            """Helper function to process a single item from the dataset"""
            street_idx, shop_idx, label = self.pairs[idx]
            
            # Load images from URLs
            street_img = self.dataset_imgs[street_idx]['street_photo_image']
            shop_img = self.dataset_imgs[shop_idx]['shop_photo_image']
            
            # Ensure images are in RGB format
            if street_img.mode != 'RGB':
                street_img = street_img.convert('RGB')
            if shop_img.mode != 'RGB':
                shop_img = shop_img.convert('RGB')
            
            # Apply ToTensor and Normalize transformation
            street_img = to_tensor_and_normalize(street_img)
            shop_img = to_tensor_and_normalize(shop_img)
            
            return street_img, shop_img, label

        # Handle both single index and batch (list) cases
        if isinstance(idx, list):
            # Process batch
            batch_items = [_process_single_item(i) for i in idx]
            street_imgs, shop_imgs, labels = zip(*batch_items)
            
            return (torch.stack(street_imgs), 
                   torch.stack(shop_imgs),
                   torch.tensor(labels))
        
        # Process single item
        return _process_single_item(idx)

if __name__ == '__main__':
    dataset = Street2ShopImageSimilarityDataset(ratio=0.5)
    print(len(dataset))
    print(dataset[0])