def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.lr)
