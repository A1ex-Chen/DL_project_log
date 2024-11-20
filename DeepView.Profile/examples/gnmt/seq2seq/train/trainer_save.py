def save(self, identifier=None, is_best=False, save_all=False):
    """
        Stores checkpoint to a file.

        :param identifier: identifier for periodic checkpoint
        :param is_best: if True stores checkpoint to 'model_best.pth'
        :param save_all: if True stores checkpoint after completed training
            epoch
        """

    def write_checkpoint(state, filename):
        filename = os.path.join(self.save_path, filename)
        logging.info(f'Saving model to {filename}')
        torch.save(state, filename)
    if self.distributed:
        model_state = self.model.module.state_dict()
    else:
        model_state = self.model.state_dict()
    state = {'epoch': self.epoch, 'state_dict': model_state, 'optimizer':
        self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict
        (), 'loss': getattr(self, 'loss', None)}
    state = dict(list(state.items()) + list(self.save_info.items()))
    if identifier is not None:
        filename = self.checkpoint_filename % identifier
        write_checkpoint(state, filename)
    if is_best:
        filename = 'model_best.pth'
        write_checkpoint(state, filename)
    if save_all:
        filename = f'checkpoint_epoch_{self.epoch:03d}.pth'
        write_checkpoint(state, filename)
