def _progress(self, batch_idx):
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(self.data_loader, 'n_samples'):
        current = batch_idx * self.data_loader.batch_size
        total = self.data_loader.n_samples
    else:
        current = batch_idx
        total = self.len_epoch
    return base.format(current, total, 100.0 * current / total)
