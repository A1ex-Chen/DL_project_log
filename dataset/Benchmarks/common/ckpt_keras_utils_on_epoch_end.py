def on_epoch_end(self, epoch, logs=None):
    """
        Note: We immediately increment epoch
        from index-from-0 to index-from-1
        to match the TensorFlow output.
        Normally, ckpts/best is the best saved state,
              and ckpts/last is the last saved state.
        Procedure:
        1. Write current state to ckpts/work
        2. Rename ckpts/work to ckpts/epoch/NNN
        3. If best, link ckpts/best to ckpts/epoch/NNN
        4. Link ckpts/last to ckpts/epoch/NNN
        5. Clean up old ckpts according to keep policy
        """
    epoch += 1
    dir_root = PosixPath(self.ckpt_directory).resolve()
    dir_work = dir_root / 'ckpts/work'
    dir_best = dir_root / 'ckpts/best'
    dir_last = dir_root / 'ckpts/last'
    dir_epochs = dir_root / 'ckpts/epochs'
    dir_this = dir_epochs / ('%03i' % epoch)
    if not self.save_check(logs, epoch):
        return
    if os.path.exists(dir_this):
        self.debug("remove:  '%s'" % self.relpath(dir_this))
        shutil.rmtree(dir_this)
    os.makedirs(dir_epochs, exist_ok=True)
    os.makedirs(dir_work, exist_ok=True)
    self.write_model(dir_work, epoch)
    self.debug("rename:  '%s' -> '%s'" % (self.relpath(dir_work), self.
        relpath(dir_this)))
    os.rename(dir_work, dir_this)
    self.epochs.append(epoch)
    if self.epoch_best == epoch:
        self.symlink(dir_this, dir_best)
    self.symlink(dir_this, dir_last)
    self.clean(epoch)
