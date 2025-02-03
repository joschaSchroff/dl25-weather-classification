import lightning as L
from lightning import LightningModel
import torch.nn as nn
import torch



class Model(L.LightningModule):
    def __init__(self,model,learning_rate):
        super(LightningModel, self).__init__()
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