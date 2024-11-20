def delete(self, epoch):
    dir_old = 'save/ckpts/epochs/%03i' % epoch
    if os.path.exists(dir_old):
        self.debug("removing: '%s'" % dir_old)
        shutil.rmtree(dir_old)
    else:
        self.info('checkpoint for epoch=%i disappeared!' % epoch)
    self.epochs.remove(epoch)
