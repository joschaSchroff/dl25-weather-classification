from lightning import Trainer, seed_everything
from model import get_model
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from data_loading import WeatherDataModule, get_transforms, get_val_transforms
from args import TrainArgs
from simple_parsing import parse



def get_wandb_logger(args: TrainArgs):
    wandb.finish()
    wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.wandb_name, config=args)
    wandb_logger = WandbLogger()
    return wandb_logger

def get_trainer(epochs, logger, device):
    if(device == 'auto'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif(device == 'cuda'):
        assert torch.cuda.is_available(), "CUDA is not available"
    return Trainer(max_epochs=epochs, logger=logger, accelerator=device)

def train_model(model, data_module, args: TrainArgs):
    logger = get_wandb_logger(args)
    trainer = get_trainer(args.num_epochs, logger, args.accelerator)
    if args.val_before_train:
        trainer.validate(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)
    wandb.finish()


def main(args: TrainArgs):
    seed_everything(args.seed, workers=True)
    model = get_model(args.model_name, args.num_classes, args.learning_rate)
    data_module = WeatherDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, transform=get_transforms(), val_transform=get_val_transforms())
    train_model(model, data_module, args)


if __name__ == "__main__":
    config_path = "./cfgs/config.yaml"
    args = parse(TrainArgs, config_path=config_path)
    main(args)