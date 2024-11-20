def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix
    ):
    if self.wandb or self.clearml:
        files = sorted(self.save_dir.glob('val*.jpg'))
    if self.wandb:
        self.wandb.log({'Validation': [wandb.Image(str(f), caption=f.name) for
            f in files]})
    if self.clearml:
        self.clearml.log_debug_samples(files, title='Validation')
    if self.comet_logger:
        self.comet_logger.on_val_end(nt, tp, fp, p, r, f1, ap, ap50,
            ap_class, confusion_matrix)
