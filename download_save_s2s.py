from datasets import load_dataset
from tqdm import tqdm

def download_and_save_street2shop():
    print("Downloading Street2Shop dataset from HuggingFace...")
    
    try:
        # Download the dataset
        dataset = load_dataset("petr7555/street2shop")
        
        print("\nDataset downloaded successfully!")
        print(f"Dataset contains {len(dataset['train'])} training samples")
        print(f"Dataset contains {len(dataset['test'])} test samples")
        
        # Save the dataset locally
        print("\nSaving dataset locally...")
        dataset.save_to_disk("street2shop")
        print("Dataset saved successfully to 'street2shop' directory")
        
    except Exception as e:
        print(f"Error downloading/saving dataset: {e}")

if __name__ == "__main__":
    download_and_save_street2shop()
