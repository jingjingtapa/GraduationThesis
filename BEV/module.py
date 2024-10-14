import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

class LitModel(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(LitModel, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        test_loss = F.mse_loss(y_hat, y)
        self.log('test_loss', test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
