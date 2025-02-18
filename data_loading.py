import kagglehub
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import lightning as L
from typing import List, Tuple
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def download_dataset(path="./data/weather-dataset"):
    # Download latest version
    download_path = kagglehub.dataset_download("jehanbhathena/weather-dataset")
    os.makedirs(path, exist_ok=True)
    download_path = os.path.join(download_path, "dataset")
    os.rename(download_path, path)
    print("Path to dataset files:", path)

def load_image_paths_and_labels(root_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Returns:
        Tuple[List[str], List[int], List[str]]: List of image paths, corresponding labels, and class names.
    """
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    image_paths, labels = [], []
    
    for label, class_name in enumerate(classes):
        class_path = os.path.join(root_dir, class_name)
        file_names = sorted(
            [img_name for img_name in os.listdir(class_path) if img_name.endswith(".jpg")]
        )
        image_paths.extend([os.path.join(class_path, img_name) for img_name in file_names])
        labels.extend([label] * len(file_names))
    
    return image_paths, labels, classes

class WeatherDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset containing class folders.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.transform = transform
        self.image_paths, self.labels, self.classes = load_image_paths_and_labels(root_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB") 

        if self.transform:
            image = self.transform(image)

        return image, label

class WeatherDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, transform=None, val_transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.val_transform = val_transform

    def setup(self, stage=None):
        full_dataset = WeatherDataset(root_dir=self.data_dir, transform=None)

        # Split dataset into training (70%), validation (15%) and test (15%) sets
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
        self.train_dataset.dataset.transform = self.transform
        self.val_dataset.dataset.transform = self.val_transform
        self.test_dataset.dataset.transform = self.val_transform
        # class weights for train
        labels = [label for _, label in self.train_dataset]
        class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def get_class_weights(self):
        return self.class_weights


def get_transforms():
    return transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(size=(224,224), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.5,1.0),saturation=(0.5,1.5),hue=(-0.1,0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

if __name__ == "__main__":
    # path = download_dataset(path="./data/weather-dataset")
    path = "./data/weather-dataset"

    data_module = WeatherDataModule(data_dir=path, transform=get_transforms())
    data_module.setup()
    train_loader = data_module.train_dataloader()

    # Iterate over first 10 batches and print the shape of each batch
    for i, (images, labels) in enumerate(train_loader):
        if i == 10:
            break
        print(f"Batch {i}: {images.shape}")
