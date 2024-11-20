def save_checkpoint(self, val_loss, model):
    if self.verbose:
        print(
            f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
    torch.save(model.state_dict(), self.path)
    self.val_loss_min = val_loss
