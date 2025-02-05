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
    model_name: str = field(type=str, default="efficientnet")
    num_classes: int = field(type=int, default=11)

    #Data
    data_dir: str = field(type=str, default="./data")

    #Wandb
    wandb_project: str = field(type=str, default='weather-classification')
    wandb_name: str = field(type=str, default='weather-classification')
    wandb_entity: str = field(type=str, default='dl25-weather')