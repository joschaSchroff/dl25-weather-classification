import lightning as L
import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet, ResNet50_Weights
from torchmetrics.classification import Accuracy, F1Score
from sklearn.metrics import f1_score, accuracy_score



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
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        self.outputs.append((y, y_hat))
        return loss
    
    def on_validation_epoch_end(self):
        y = torch.cat([out[0] for out in self.outputs])
        y_hat = torch.cat([out[1] for out in self.outputs])
        acc = self.accuracy(y_hat, y)
        acc_sklearn = accuracy_score(y.cpu(), y_hat.cpu().argmax(dim=1))
        f1 = self.f1(y_hat, y)
        f1_sklearn = f1_score(y.cpu(), y_hat.cpu().argmax(dim=1), average="macro")
        self.log("val_f1", f1)
        self.log("val_acc", acc)
        print(f"Validation F1 Score: {f1}")
        print(f"Validation F1 Score (sklearn): {f1_sklearn}")
        print(f"Validation Accuracy: {acc}")
        print(f"Validation Accuracy (sklearn): {acc_sklearn}")
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    

def get_model(model_name, num_classes, learning_rate):
    if model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return WeatherModel(model, learning_rate, num_classes)
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
        return WeatherModel(model, learning_rate, num_classes)
    else:
        raise ValueError("Model not found")