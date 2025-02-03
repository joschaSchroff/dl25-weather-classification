from lightning import Trainer
from model import WeatherModel
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

def get_wandb_logger():
    wandb.init(entity="dl25-weather", project="weather-classification")
    wandb_logger = WandbLogger(name='weather-classification',project='weather-classification')
    return wandb_logger

def get_trainer(epochs, logger):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return Trainer(max_epochs=epochs, logger=logger, accelerator=device)

def train_model(model, train_loader, val_loader, epochs):
    logger = get_wandb_logger()
    trainer = get_trainer(epochs, logger)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    model = WeatherModel()
    train_model(model, None, None, 10)