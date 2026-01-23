import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.classification import Accuracy

class Classifier(LightningModule):
    def __init__(self, latent_dim: int, num_classes: int, hidden_dim: int = 256, lr: float = 1e-3, dropout: float = 0.2,
                 select_factor: int=0):
        super().__init__()
        self.save_hyperparameters()

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(y_pred, y_true)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y[:, self.hparams.select_factor])
        acc = self.train_acc(logits, y[:, self.hparams.select_factor])

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y[:, self.hparams.select_factor])
        acc = self.val_acc(logits, y[:, self.hparams.select_factor])

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y[:, self.hparams.select_factor])
        acc = self.test_acc(logits, y[:, self.hparams.select_factor])

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
