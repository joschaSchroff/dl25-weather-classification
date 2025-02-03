import kagglehub
import torch
import torch.nn as nn


def download_dataset():
    # Download latest version
    path = kagglehub.dataset_download("jehanbhathena/weather-dataset")
    print("Path to dataset files:", path)


if __name__ == "__main__":
    #download_dataset()
    pass