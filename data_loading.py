import kagglehub

# Download latest version
path = kagglehub.dataset_download("jehanbhathena/weather-dataset", path="data")

print("Path to dataset files:", path)