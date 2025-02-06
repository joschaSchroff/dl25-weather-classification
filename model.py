import lightning as L
import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights, EfficientNet_B1_Weights
from torchvision.models.mobilenet import MobileNet_V2_Weights
from torchvision.models.vision_transformer import ViT_B_16_Weights
from torchvision.models.swin_transformer import Swin_T_Weights
from torchmetrics.classification import Accuracy, F1Score

class WeatherModel(L.LightningModule):
    def __init__(self,model,learning_rate,num_classes):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.outputs = []
    
    def forward(self,x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.accuracy.update(y_hat, y)
        self.f1.update(y_hat, y)
        return loss
    
    def on_validation_epoch_end(self):
        acc = self.accuracy.compute()
        f1 = self.f1.compute()

        self.log("val_acc", acc)
        self.log("val_f1", f1)

        self.accuracy.reset()
        self.f1.reset()
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    

def get_model(model_name, num_classes, learning_rate):
    if model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return WeatherModel(model, learning_rate, num_classes)
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return WeatherModel(model, learning_rate, num_classes)
    elif model_name == "efficientnetb1":
        model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return WeatherModel(model, learning_rate, num_classes)
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return WeatherModel(model, learning_rate, num_classes)
    elif model_name == "vit":
        model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads[0] = nn.Linear(model.heads[0].in_features, num_classes)
        return WeatherModel(model, learning_rate, num_classes)
    elif model_name == "swin":
        model = models.swin_t(Swin_T_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return WeatherModel(model, learning_rate, num_classes)
    else:
        raise ValueError("Model not found")