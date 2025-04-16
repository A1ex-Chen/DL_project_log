def save_model(self):
    """Save model training checkpoints with additional metadata."""
    import io
    import pandas as pd
    buffer = io.BytesIO()
    torch.save({'epoch': self.epoch, 'best_fitness': self.best_fitness,
        'model': None, 'ema': deepcopy(self.ema.ema).half(), 'updates':
        self.ema.updates, 'optimizer': convert_optimizer_state_dict_to_fp16
        (deepcopy(self.optimizer.state_dict())), 'train_args': vars(self.
        args), 'train_metrics': {**self.metrics, **{'fitness': self.fitness
        }}, 'train_results': {k.strip(): v for k, v in pd.read_csv(self.csv
        ).to_dict(orient='list').items()}, 'date': datetime.now().isoformat
        (), 'version': __version__, 'license':
        'AGPL-3.0 (https://ultralytics.com/license)', 'docs':
        'https://docs.ultralytics.com'}, buffer)
    serialized_ckpt = buffer.getvalue()
    self.last.write_bytes(serialized_ckpt)
    if self.best_fitness == self.fitness:
        self.best.write_bytes(serialized_ckpt)
    if (self.save_period > 0 and self.epoch > 0 and self.epoch % self.
        save_period == 0):
        (self.wdir / f'epoch{self.epoch}.pt').write_bytes(serialized_ckpt)
