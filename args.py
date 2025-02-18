from dataclasses import dataclass
from simple_parsing import field

@dataclass(kw_only=True)
class TrainArgs:
    # Trainer
    num_workers: int = field(type=int, default=4)
    accelerator: str = field(type=str, default='auto')
    seed: int = field(type=int, default=42)
    num_epochs: int = field(type=int, default=5)
    batch_size: int = field(type=int, default=32)
    learning_rate: float = field(type=float, default=0.001)
    lr_step_size: int = field(type=int, default=50)
    lr_gamma: float = field(type=float, default=0.1)
    model_name: str = field(type=str, default="efficientnet")
    num_classes: int = field(type=int, default=11)
    val_before_train: bool = field(type=bool, default=True)

    early_stopping: bool = field(type=bool, default=True)
    early_stopping_patience: int = field(type=int, default=3)
    early_stopping_metric: str = field(type=str, default="val_loss")
    early_stopping_mode: str = field(type=str, default="min")
    early_stopping_delta: float = field(type=float, default=0.001)

    #Data
    data_dir: str = field(type=str, default="./data")
    use_class_weights: bool = field(type=bool, default=True)
    save_model: bool = field(type=bool, default=False)
    save_model_path: str = field(type=str, default="./models")

    #Wandb
    wandb_project: str = field(type=str, default='weather-classification')
    wandb_name: str = field(type=str, default='weather-classification')
    wandb_entity: str = field(type=str, default='dl25-weather')