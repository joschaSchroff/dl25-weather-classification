import lightning as L
import torch.nn as nn
import torch
import torchvision.models as models



class WeatherModel(L.LightningModule):
    def __init__(self,model,learning_rate):
        super(L.LightningModule, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
    
    def forward(self,x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss
    
    def validation_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    

def get_model(model_name, num_classes, learning_rate):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return WeatherModel(model, learning_rate)
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return WeatherModel(model, learning_rate)
    else:
        raise ValueError("Model not found")