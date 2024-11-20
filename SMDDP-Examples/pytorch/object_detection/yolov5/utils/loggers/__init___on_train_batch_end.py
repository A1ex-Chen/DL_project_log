def on_train_batch_end(self, ni, model, imgs, targets, paths, plots):
    if plots:
        if ni == 0:
            if self.tb and not self.opt.sync_bn:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self.tb.add_graph(torch.jit.trace(de_parallel(model),
                        imgs[0:1], strict=False), [])
        if ni < 3:
            f = self.save_dir / f'train_batch{ni}.jpg'
            plot_images(imgs, targets, paths, f)
        if (self.wandb or self.clearml) and ni == 10:
            files = sorted(self.save_dir.glob('train*.jpg'))
            if self.wandb:
                self.wandb.log({'Mosaics': [wandb.Image(str(f), caption=f.
                    name) for f in files if f.exists()]})
            if self.clearml:
                self.clearml.log_debug_samples(files, title='Mosaics')
