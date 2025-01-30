from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset
import random
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
from datasets import Dataset
import faiss
import torch
import os

from models.xception import XceptionModel
SEED = 42


class Street2ShopImageSimilarityDataset(Dataset):
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
    
    to_tensor_and_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    def __init__(self, num_negative_pairs=2, ratio=1.0):
        print('Initializing Street2ShopImageSimilarityDataset..')
        
        disk_path = f'street2shop_{ratio}' if ratio < 1 else 'street2shop'
        hf_path = f'petr7555/street2shop'
        self.num_negative_pairs = num_negative_pairs
        sampled_indices = None
        
        # Load the dataset from disk, remove the image columns to save memory
        try:
            print(f"Loading dataset from {disk_path}...")
            
            self.dataset = load_from_disk(disk_path)
            
            # in case we load the original street2shop dataset
            if 'index' not in self.dataset.column_names:
                self.dataset = self.dataset['train'].remove_columns(['type', 'street_photo_image', 'shop_photo_image'])
                self.dataset = self.dataset.add_column('index', list(range(len(self.dataset))))
        except FileNotFoundError:
            print(f"Dataset not found at {disk_path}. Loading {hf_path} dataset...")
            
            try:
                self.dataset = load_from_disk(hf_path.split('/')[-1])['train'].remove_columns(['type', 'street_photo_image', 'shop_photo_image'])
            except FileNotFoundError:
                self.dataset = load_dataset(hf_path)
                self.dataset.save_to_disk(hf_path.split('/')[-1])
                self.dataset = self.dataset['train'].remove_columns(['type', 'street_photo_image', 'shop_photo_image'])
        
            # Add an index column to the dataset
            self.dataset = self.dataset.add_column('index', list(range(len(self.dataset))))
            self._sample_dataset(ratio) # dataset already split/sampled & updated inside the function
            
            self.dataset.save_to_disk(disk_path)
        
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
        
        disk_path = f'street2shop_imgs_{ratio}' if ratio < 1 else 'street2shop_imgs'
        
        # Load dataset images
        valid_idx_set = set()
        try:
            print(f"Loading dataset images from {disk_path}...")
            
            self.dataset_imgs = load_from_disk(disk_path)
            
            valid_idx_set = set(self.dataset_imgs['index'])
        except FileNotFoundError:
            print(f"Dataset images not found at {disk_path}. Loading {hf_path} dataset images and performing transforms...")
            
            try:
                self.dataset_imgs = load_from_disk(hf_path.split('/')[-1])['train'].select_columns(['street_photo_image', 'shop_photo_image', 'left', 'top', 'width', 'height'])
            except FileNotFoundError:
                self.dataset_imgs = load_dataset(hf_path)
                self.dataset_imgs.save_to_disk(hf_path.split('/')[-1])
                self.dataset_imgs = self.dataset_imgs['train'].select_columns(['street_photo_image', 'shop_photo_image', 'left', 'top', 'width', 'height'])
            
            # Add an index column to the dataset
            self.dataset_imgs = self.dataset_imgs.add_column('index', list(range(len(self.dataset_imgs))))
            self.dataset_imgs = self.dataset_imgs.select(sampled_indices) # this matches the sampled indices of the dataset - pairs are generated based on this
            
            # now to remove the corrupted images, and images that are giving errors while transforming
            def _identify_valid_rows(dataset_imgs):
                valid_idx_set = set()
                under_process_idx = []
                for i in tqdm(range(len(dataset_imgs)), desc="Identifying valid images"):
                    try:
                        item = dataset_imgs[i]            
                        valid_idx_set.add(item['index'])
                        under_process_idx.append(i) # If no exception is raised, add the index to valid_idx
                    except Exception as e:
                        print(f"Error processing image at index {i}: {e}")
                        continue  # Skip the corrupted image
                return valid_idx_set, under_process_idx
            
            valid_idx_set, under_process_idx = _identify_valid_rows(self.dataset_imgs)
            self.dataset_imgs = self.dataset_imgs.select(under_process_idx)
            
            # Apply transformations with tqdm
            self.dataset_imgs = self.dataset_imgs.map(
                lambda example: self._transform_image(example),
                desc="Transforming images"
            )
            
            # Rows with further issues while transforming .. to be subtracted from valid_idx
            invalid_idx_set = set([item['index'] for item in self.dataset_imgs if not item['valid']])
            valid_idx_set = valid_idx_set - invalid_idx_set # subtracting..
            
            # finally removing the problematic rows found during transforms
            self.dataset_imgs = self.dataset_imgs.filter(lambda x: x['valid'])
            self.dataset_imgs.save_to_disk(disk_path)
        
        print('# of valid idx set:', len(valid_idx_set))
        
        # Filter pairs to only include valid indices
        self.pairs = [(s_idx, sh_idx, label) for s_idx, sh_idx, label in self.pairs if s_idx in valid_idx_set and sh_idx in valid_idx_set]
        print('# of pairs after filtering:', len(self.pairs))
    
        
    def _sample_dataset(self, ratio):
        if ratio == 1:
            return
        
        df = self.dataset.to_pandas()
        train_df, _ = train_test_split(
            df,
            test_size=1-ratio,
            stratify=df['category'],
            random_state=SEED
        )   # Perform stratified split
        self.dataset = Dataset.from_pandas(train_df)    # Convert back to a Hugging Face dataset

    
    def _get_cropped_image(self, item):
        street_img = item['street_photo_image']
        left = int(item['left'])
        top = int(item['top'])
        width = int(item['width'])
        height = int(item['height'])
        return street_img.crop((left, top, left + width, top + height))
    

    def _transform_image(self, example):
        try:
            if example['street_photo_image'].mode != 'RGB':
                example['street_photo_image'] = example['street_photo_image'].convert('RGB')
            example['street_photo_image'] = self._get_cropped_image(example)
            example['street_photo_image'] = self.street_transform(example['street_photo_image'])
            
            if example['shop_photo_image'].mode != 'RGB':
                example['shop_photo_image'] = example['shop_photo_image'].convert('RGB')
            example['shop_photo_image'] = self.shop_transform(example['shop_photo_image'])
            
            # Mark the example as valid
            return {**example, 'valid': True}
        except Exception as e:
            print(f"Error processing image: {e}")
            # Mark the example as invalid
            return {**example, 'valid': False}

           
    def _generate_pairs(self):
        pairs = []
        
        for idx, item in tqdm(enumerate(self.dataset), desc="Generating pairs", total=len(self.dataset)):
            # Positive pair
            pairs.append((idx, idx, 1))
            # Generate negative pairs
            neg_indices = set()
            while len(neg_indices) < self.num_negative_pairs:
                neg_idx = self._get_negative_pair(item)
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


    def _get_negative_pair(self, item):
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
            street_img = self.to_tensor_and_normalize(street_img)
            shop_img = self.to_tensor_and_normalize(shop_img)
            
            return street_img, shop_img, label

        # If idx is a single integer, return a single tuple
        if isinstance(idx, int):
            return _process_single_item(idx)
        
        output = []
        for i in idx:
            try:
                output.append(_process_single_item(i))
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue
        
        return output


################################################################ 


class Street2ShopImageSimilarityTestDataset(Dataset):
    def __init__(self, feature_extractor, batch_size=128, ratio=1.0, save_dir='.'):
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        
        self.index_file = f's2s_test_{ratio}_{feature_extractor.model_name}{feature_extractor.embedding_dim}.faiss' if ratio < 1 else f's2s_test_{feature_extractor.model_name}{feature_extractor.embedding_dim}.faiss'
        self.dataset_path = f'street2shop_test_{ratio}' if ratio < 1 else 'street2shop_test'
        
        # load and sample/shuffle the test dataset
        try:
            self.test_dataset = load_from_disk(self.dataset_path)
        except FileNotFoundError:
            try:
                self.test_dataset = load_from_disk('street2shop')['test'].remove_columns(['street_photo_image', 'shop_photo_image'])
            except FileNotFoundError:
                self.test_dataset = load_dataset('street2shop')
                self.test_dataset.save_to_disk('street2shop')
                self.test_dataset = self.test_dataset['test'].remove_columns(['street_photo_image', 'shop_photo_image'])

            def _sample_dataset(dataset, ratio):
                valid_indices = []
                for i in random.sample(range(len(dataset)), int(len(dataset) * ratio)): # random indices
                    try:
                        item = dataset[i]   # check if the image at the index is uncorrupted
                        valid_indices.append(i)
                    except Exception as e:
                        print(f"Error processing image at index {i}: {e}")
                        continue
                return dataset.select(valid_indices)
            self.test_dataset = _sample_dataset(self.test_dataset, ratio)
            self.test_dataset = self.test_dataset.add_column('index', list(range(len(self.test_dataset))))
            self.test_dataset.save_to_disk(self.dataset_path)
        
        # transform and index shop images as vectors in FAISS local
        shop_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        res = faiss.StandardGpuResources()
        if os.path.exists(self.index_file):
            print(f"Loading FAISS index from {self.index_file}...")
            self.gpu_index = faiss.read_index(os.path.join(save_dir, self.index_file))
            self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.gpu_index)  # Transfer to GPU
        else:
            # Create an HNSW index for inner product (cosine similarity)
            index = faiss.IndexHNSWFlat(self.feature_extractor.embedding_dim, 32)  # 32 is the number of neighbors in the graph
            self.gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Transfer to GPU
            
            # Process images in batches
            for start_idx in tqdm(range(0, len(self.test_dataset), batch_size), desc="Indexing shop images"):
                end_idx = min(start_idx + batch_size, len(self.test_dataset))
                batch_items = [self.test_dataset[i] for i in range(start_idx, end_idx)]
                
                # Transform and stack images
                shop_imgs = torch.stack([shop_transform(item['shop_photo_image']) for item in batch_items])
                
                with torch.no_grad():
                    shop_imgs = shop_imgs.to(self.feature_extractor.device)
                    features = self.feature_extractor(shop_imgs).cpu().numpy()
                    # Normalize the features for cosine similarity
                    if not self.feature_extractor.normalization:
                        faiss.normalize_L2(features)
                
                self.gpu_index.add(features)
            
            # Save the index to a file
            print(f"Saving FAISS index to {self.index_file}...")
            faiss.write_index(faiss.index_gpu_to_cpu(self.gpu_index), os.path.join(save_dir, self.index_file))
        
    def __len__(self):
        return len(self.test_dataset)

    def __getitem__(self, idx):
        return self.test_dataset[idx]
    
    def search(self, street_photo, k=5):
        """Search for the top-k similar shop images to the street photo."""
        street_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        street_img = street_transform(street_photo).unsqueeze(0)
        with torch.no_grad():
            street_img = street_img.to(self.feature_extractor.device)
            query_features = self.feature_extractor(street_img).cpu().numpy()
            # Normalize the query features
            if not self.feature_extractor.normalization:
                faiss.normalize_L2(query_features)
        
        distances, indices = self.gpu_index.search(query_features, k)
        return indices.tolist()


################################################################ 


def evaluate_top_k_accuracies(dataset, k=10, num_samples=100, num_visualize=5):
    """
    Evaluate the model using top-1, top-3, top-5 and top-10 accuracies.
    
    :param dataset: An instance of Street2ShopImageSimilarityTestDataset
    :param num_samples: The number of random samples to evaluate
    :param num_visualize: The number of random samples to return for visualization
    :return: A tuple containing:
        - Dictionary with top-1, top-3, top-5 and top-10 accuracies
        - List of tuples (street_idx, retrieved_indices) for visualization
    """
    # Randomly select indices from the test dataset
    indices = random.sample(range(len(dataset.test_dataset)), num_samples)
    
    top_1_count = 0
    top_3_count = 0
    top_5_count = 0
    top_10_count = 0
    
    # Store all evaluation results
    all_results = []
    
    for idx in indices:
        # Get the street photo and its true category
        item = dataset.test_dataset[idx]
        street_photo = item['street_photo_image']
        true_idx = item['index']
        
        # Perform the search
        retrieved_indices = dataset.search(street_photo, k=k)
        
        # Store result
        all_results.append((idx, retrieved_indices[0]))
        
        # Check if the true category is in the top-k results
        for position, retrieved_idx in enumerate(retrieved_indices[0]):  # retrieved_indices is a 2D array
            retrieved_item = dataset.test_dataset[retrieved_idx]
            if retrieved_item['index'] == true_idx:
                if position == 0:
                    top_1_count += 1
                if position < 3:
                    top_3_count += 1
                if position < 5:
                    top_5_count += 1
                if position < 10:
                    top_10_count += 1
                break  # Stop checking once the correct category is found
    
    accuracies = {
        'top_1_accuracy': top_1_count / num_samples,
        'top_3_accuracy': top_3_count / num_samples,
        'top_5_accuracy': top_5_count / num_samples,
        'top_10_accuracy': top_10_count / num_samples
    }
    
    # Randomly select m samples for visualization
    visualization_data = random.sample(all_results, min(num_visualize, len(all_results)))
    
    return accuracies, visualization_data


################################################################ 


if __name__ == '__main__':
    dataset = Street2ShopImageSimilarityDataset(ratio=0.05)
    print(len(dataset))
    print(dataset[0])
    del dataset

    # model = XceptionModel(embedding_dim=512)
    # test_dataset = Street2ShopImageSimilarityTestDataset(model, ratio=0.6)
    # print(len(test_dataset))
    # print(test_dataset[0])

    # # Assuming `dataset` is an instance of Street2ShopImageSimilarityTestDataset
    # accuracies = evaluate_top_k_accuracies(test_dataset, num_samples=10)
    # print("Top-1 Accuracy:", accuracies['top_1_accuracy'])
    # print("Top-3 Accuracy:", accuracies['top_3_accuracy'])
    # print("Top-5 Accuracy:", accuracies['top_5_accuracy'])