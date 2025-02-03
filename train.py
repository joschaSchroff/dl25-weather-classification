from lightning import Trainer
from model import get_model
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from data_loading import WeatherDataModule, get_transforms



def get_wandb_logger():
    wandb.init(entity="dl25-weather", project="weather-classification")
    wandb_logger = WandbLogger(name='weather-classification', project='weather-classification')
    return wandb_logger

def get_trainer(epochs, logger):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return Trainer(max_epochs=epochs, logger=logger, accelerator=device)

def train_model(model, data_module, epochs):
    logger = get_wandb_logger()
    trainer = get_trainer(epochs, logger)
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    model = get_model("efficientnet", 11, 0.001)
    data_module = WeatherDataModule(data_dir="/home/joscha/deep-learning/weather-dataset/3/dataset", transform=get_transforms())
    train_model(model, data_module, 5)