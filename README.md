# Shopping Image Search

A deep learning-based image similarity search system that matches street photos with shop photos using the Street2Shop dataset. The system learns embeddings for both street and shop photos and uses FAISS for efficient similarity search.

## Overview

This project implements an image similarity search system for fashion items. Given a photo of clothing "in the wild" (street photo), it can find similar items from an online shopping catalog (shop photos). The system uses deep learning models to generate embeddings for images and FAISS for fast similarity search.

## Features

- Support for multiple deep learning architectures:
  - Xception network
  - CLIP ViT (Vision Transformer)
- Efficient similarity search using FAISS
- Data augmentation pipeline for both street and shop photos
- Training with contrastive learning using cosine embedding loss
- Evaluation metrics including top-k accuracy (k=1,3,5)
- GPU acceleration support
- Configurable dataset sampling ratio for experiments

## Installation
```bash
git clone https://github.com/petr7555/shopping-image-search.git
cd shopping-image-search
pip install -r requirements.txt
```

## Dataset

The project uses the Street2Shop dataset, which contains pairs of street photos and their matching shop photos. The dataset is automatically downloaded from Hugging Face's dataset hub.

Key dataset features:
- Automatic handling of corrupted images
- Data augmentation specific to street and shop photos
- Support for dataset sampling for quick experiments
- Stratified splitting for training/validation

## Project Structure
```
shopping-image-search/
├── dataset.py # Dataset classes and data loading utilities
├── train.py # Training loop and validation
├── models/
│ ├── base.py # Base model class
│ ├── xception.py # Xception model implementation
│ └── clip_vit.py # CLIP ViT model implementation
```

## Usage

### Training

```python
from models.xception import XceptionModel
from models.clip_vit import CLIPViTModel
from dataset import Street2ShopImageSimilarityDataset
from train import train

# Initialize dataset
dataset = Street2ShopImageSimilarityDataset(ratio=0.6) # Use 60% of data

# Choose a model
model = XceptionModel(embedding_dim=512)
# OR
model = CLIPViTModel(embedding_dim=512)

# Train the model
train(model, train_loader, optimizer, val_loader, device='cuda', save_dir='saved_models')
```

### Inference

```python
from dataset import Street2ShopImageSimilarityTestDataset

# Initialize test dataset with trained model
test_dataset = Street2ShopImageSimilarityTestDataset(model, ratio=0.6)

# Search similar items
similar_indices = test_dataset.search(street_photo, k=5)
```

## Model Architecture

### Base Model
- Defines common functionality for all models
- Handles model saving/loading
- Implements prediction interface

### Xception Model
- Based on the Xception architecture
- Pretrained on ImageNet
- Modified final layer for embedding generation

### CLIP ViT Model
- Uses OpenAI's CLIP Vision Transformer
- Leverages pretrained visual understanding
- Projects embeddings to specified dimension

## Training Details

- Uses contrastive learning with cosine embedding loss
- Implements early stopping based on validation loss
- Supports both CPU and GPU training
- Includes memory optimization for large datasets
- Uses data augmentation specific to street and shop photos:
  - Street photos: More aggressive augmentation (blur, affine transforms)
  - Shop photos: Lighter augmentation (color jitter, flips)

## Evaluation

The system evaluates performance using:
- Top-1, Top-3, and Top-5 accuracy
- Validation loss
- Cosine similarity metrics

## Acknowledgments

- Street2Shop dataset
- CLIP model from OpenAI
- FAISS from Facebook Research
