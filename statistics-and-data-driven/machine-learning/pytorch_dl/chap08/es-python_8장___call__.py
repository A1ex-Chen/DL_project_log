def __call__(self, val_loss, model):
    score = -val_loss
    if self.best_score is None:
        self.best_score = score
        self.save_checkpoint(val_loss, model)
    elif score < self.best_score + self.delta:
        self.counter += 1
        print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        if self.counter >= self.patience:
            self.early_stop = True
    else:
        self.best_score = score
        self.save_checkpoint(val_loss, model)
        self.counter = 0
