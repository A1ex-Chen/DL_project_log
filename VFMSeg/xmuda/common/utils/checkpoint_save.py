def save(self, name, tag=True, **kwargs):
    if not self.save_dir:
        return
    data = dict()
    if isinstance(self.model, (DataParallel, DistributedDataParallel)):
        data['model'] = self.model.module.state_dict()
    else:
        data['model'] = self.model.state_dict()
    if self.optimizer is not None:
        data['optimizer'] = self.optimizer.state_dict()
    if self.scheduler is not None:
        data['scheduler'] = self.scheduler.state_dict()
    data.update(kwargs)
    save_file = os.path.join(self.save_dir, '{}.pth'.format(name))
    self._print('Saving checkpoint to {}'.format(os.path.abspath(save_file)))
    torch.save(data, save_file)
    if tag:
        self.tag_last_checkpoint(save_file)
