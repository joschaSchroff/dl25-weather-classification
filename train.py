from lightning import Trainer, seed_everything
from model import get_model
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from data_loading import WeatherDataModule, get_transforms, get_val_transforms
from args import TrainArgs
from simple_parsing import parse
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor



def get_wandb_logger(args: TrainArgs):
    wandb.finish()
    wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.wandb_name, config=args)
    wandb_logger = WandbLogger()
    return wandb_logger

def get_trainer(epochs, logger, device, callbacks):
    if(device == 'auto'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif(device == 'cuda'):
        assert torch.cuda.is_available(), "CUDA is not available"
    return Trainer(max_epochs=epochs, logger=logger, accelerator=device, callbacks=callbacks)

def get_callbacks(args: TrainArgs):
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]
    if(args.early_stopping):
        early_stopping = EarlyStopping(monitor=args.early_stopping_metric, patience=args.early_stopping_patience, mode=args.early_stopping_mode, min_delta=args.early_stopping_delta)
        callbacks.append(early_stopping)
    return callbacks
    
def train_model(model, data_module, args: TrainArgs):
    logger = get_wandb_logger(args)
    callbacks = get_callbacks(args)
    trainer = get_trainer(args.num_epochs, logger, args.accelerator, callbacks)
    if args.val_before_train:
        trainer.validate(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)
    wandb.finish()


def main(args: TrainArgs):
    seed_everything(args.seed, workers=True)
    model = get_model(args.model_name, args.num_classes, args.learning_rate, args.lr_step_size, args.lr_gamma)
    data_module = WeatherDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, transform=get_transforms(), val_transform=get_val_transforms())
    train_model(model, data_module, args)


if __name__ == "__main__":
    config_path = "./cfgs/config.yaml"
    args = parse(TrainArgs, config_path=config_path)
    main(args)