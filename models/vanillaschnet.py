class CustomSchNet(pl.LightningModule):
    def __init__(self):
        super(CustomSchNet, self).__init__()
        self.model = SchNet()

    def forward(self, sample):
        return self.model(sample.z, sample.pos, sample.batch)

    def mse(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    def training_step(self, train_batch, batch_idx):
        logits = self.forward(train_batch).reshape(-1)
        loss = self.mse(logits, train_batch.y[:, 7])

        self.log("train_loss", loss, batch_size=32)
        return loss

    def validation_step(self, val_batch, batch_idx):
        logits = self.forward(val_batch).reshape(-1)
        loss = self.mse(logits, val_batch.y[:, 7])
        self.log("val_loss", loss, batch_size=32)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer, 100000, 0.96
        )  # torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        return [optimizer], [sched]
