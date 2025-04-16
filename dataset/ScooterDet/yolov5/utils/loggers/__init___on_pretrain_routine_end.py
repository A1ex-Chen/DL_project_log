def on_pretrain_routine_end(self, labels, names):
    if self.plots:
        plot_labels(labels, names, self.save_dir)
        paths = self.save_dir.glob('*labels*.jpg')
        if self.wandb:
            self.wandb.log({'Labels': [wandb.Image(str(x), caption=x.name) for
                x in paths]})
        if self.comet_logger:
            self.comet_logger.on_pretrain_routine_end(paths)
