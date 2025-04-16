def on_train_batch_end(self, model, ni, imgs, targets, paths, vals):
    log_dict = dict(zip(self.keys[:3], vals))
    if self.plots:
        if ni < 3:
            f = self.save_dir / f'train_batch{ni}.jpg'
            plot_images(imgs, targets, paths, f)
            if ni == 0 and self.tb and not self.opt.sync_bn:
                log_tensorboard_graph(self.tb, model, imgsz=(self.opt.imgsz,
                    self.opt.imgsz))
        if ni == 10 and (self.wandb or self.clearml):
            files = sorted(self.save_dir.glob('train*.jpg'))
            if self.wandb:
                self.wandb.log({'Mosaics': [wandb.Image(str(f), caption=f.
                    name) for f in files if f.exists()]})
            if self.clearml:
                self.clearml.log_debug_samples(files, title='Mosaics')
    if self.comet_logger:
        self.comet_logger.on_train_batch_end(log_dict, step=ni)
