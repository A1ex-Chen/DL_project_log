def on_val_end(self):
    if self.wandb or self.clearml:
        files = sorted(self.save_dir.glob('val*.jpg'))
        if self.wandb:
            self.wandb.log({'Validation': [wandb.Image(str(f), caption=f.
                name) for f in files]})
        if self.clearml:
            self.clearml.log_debug_samples(files, title='Validation')
