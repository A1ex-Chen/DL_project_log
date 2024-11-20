def log_images(self, files, name='Images', epoch=0):
    files = [Path(f) for f in (files if isinstance(files, (tuple, list)) else
        [files])]
    files = [f for f in files if f.exists()]
    if self.tb:
        for f in files:
            self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch,
                dataformats='HWC')
    if self.wandb:
        self.wandb.log({name: [wandb.Image(str(f), caption=f.name) for f in
            files]}, step=epoch)
