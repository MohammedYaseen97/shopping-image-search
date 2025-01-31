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
- Evaluation metrics including top-k accuracy (k=1,3,5,10)
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
street_photo = ...  # Load or provide the street photo
similar_indices = test_dataset.search(street_photo, k=5)
```

## Evaluation

### Evaluating Top-k Accuracies

To evaluate the model's performance using top-k accuracies, use the `evaluate_top_k_accuracies` function:

```python
from dataset import evaluate_top_k_accuracies

# Evaluate top-k accuracies
query_indices = ...  # List of indices to query
vis_indices = ...  # List of indices for visualization
accuracies, visualization_data = evaluate_top_k_accuracies(test_dataset, query_indices, vis_indices, k=10)

# Print accuracies
print("Top-1 Accuracy:", accuracies['top_1_accuracy'])
print("Top-3 Accuracy:", accuracies['top_3_accuracy'])
print("Top-5 Accuracy:", accuracies['top_5_accuracy'])
print("Top-10 Accuracy:", accuracies['top_10_accuracy'])
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
- Top-1, Top-3, Top-5 and Top-10 accuracy
- Validation loss
- Cosine similarity metrics

## Performance Evaluation

### Retrieval Performance

The performance of the image retrieval system is evaluated using top-k accuracy metrics. We compare the performance of the Xception and CLIP ViT models in terms of top-1, top-3, top-5, and top-10 accuracy.

### Visualization of Results

The system includes functionality to visualize the retrieval results. For a given query street photo, the top retrieved shop photos are displayed to assess the model's performance qualitatively.

### Model Performance Comparison

A bar chart is generated to compare the top-k accuracy of the Xception and CLIP ViT models, providing a visual representation of their performance differences.

### Enhancements to Retrieval Performance

Inspired by Pinterest's image search pipeline, the following enhancements are implemented:

1. **Multi-Crop Detection (YOLOv8):** Detects multiple crops in an image to improve retrieval accuracy.
2. **Query Expansion (BLIP-2):** Generates text captions for images to enhance query representation.
3. **Multi-Modal Embeddings (CLIP + Xception + FashionBERT):** Combines embeddings from multiple models for a richer feature representation.
4. **Multi-Stage Retrieval (Coarse-to-Fine):** A retrieval pipeline that filters candidates using coarse similarity measures before fine-tuning with more detailed models.
5. **Re-Ranking (DeiT-Small Transformer):** Refines the final ranking of retrieved images using a transformer model.

### Pipeline Execution

The full pipeline is executed to perform a fashion search, leveraging the above enhancements to improve retrieval performance.

## Acknowledgments

- Street2Shop dataset
- CLIP model from OpenAI
- FAISS from Facebook Research

## Model Selection

### Xception Model

The Xception model is chosen for its ability to capture multiple contexts through its architecture, which uses depthwise separable convolutions. This allows the model to focus on subtle details in fashion images by processing different subsets of channels separately, making it particularly effective for distinguishing fine-grained features in clothing items.

**Citation:**
- Chollet, F. (2017). [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357). In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

### CLIP ViT Model

The CLIP ViT model is based on the Vision Transformer (ViT) architecture, which divides images into patches and processes them in parallel. This patch-based approach enables the model to encode detailed information effectively, making it well-suited for fashion image retrieval where capturing intricate patterns and textures is crucial.

**Citation:**
- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). In Proceedings of the International Conference on Machine Learning (ICML).

### Why These Models?

- **Xception**: Utilizes multiple contexts through its architecture, making it adept at capturing subtle details in fashion images.
- **CLIP ViT**: Employs a patch-based approach to encode detailed information, ideal for capturing intricate patterns and textures in fashion items.
