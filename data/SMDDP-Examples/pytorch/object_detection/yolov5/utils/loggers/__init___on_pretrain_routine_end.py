def on_pretrain_routine_end(self):
    paths = self.save_dir.glob('*labels*.jpg')
    if self.wandb:
        self.wandb.log({'Labels': [wandb.Image(str(x), caption=x.name) for
            x in paths]})
    if self.clearml:
        pass
