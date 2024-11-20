def _save_checkpoint(self, epoch, save_best=False):
    """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
    arch = type(self.model).__name__
    state = {'arch': arch, 'epoch': epoch, 'state_dict': self.model.
        state_dict(), 'optimizer': self.optimizer.state_dict(),
        'monitor_best': self.mnt_best, 'config': self.config}
    filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch)
        )
    torch.save(state, filename)
    self.logger.info('Saving checkpoint: {} ...'.format(filename))
    if save_best:
        best_path = str(self.checkpoint_dir / 'model_best.pth')
        torch.save(state, best_path)
        self.logger.info('Saving current best: model_best.pth ...')
