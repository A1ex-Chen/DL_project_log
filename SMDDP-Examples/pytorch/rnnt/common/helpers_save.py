def save(self, model, ema_model, optimizer, epoch, step, best_wer, is_best=
    False):
    """Saves model checkpoint for inference/resuming training.

        Args:
            model: the model, optionally wrapped by DistributedDataParallel
            ema_model: model with averaged weights, can be None
            optimizer: optimizer
            epoch (int): epoch during which the model is saved
            step (int): number of steps since beginning of training
            best_wer (float): lowest recorded WER on the dev set
            is_best (bool, optional): set name of checkpoint to 'best'
                and overwrite the previous one
        """
    rank = 0
    if dist.is_initialized():
        dist.barrier()
        rank = dist.get_rank()
    if rank != 0:
        return
    if not is_best and epoch in self.tracked:
        return
    unwrap_ddp = lambda model: getattr(model, 'module', model)
    state = {'epoch': epoch, 'step': step, 'best_wer': best_wer,
        'state_dict': unwrap_ddp(model).state_dict(), 'ema_state_dict': 
        unwrap_ddp(ema_model).state_dict() if ema_model is not None else
        None, 'optimizer': optimizer.state_dict(), 'amp': amp.state_dict() if
        self.use_amp else None}
    if is_best:
        fpath = os.path.join(self.save_dir,
            f'{self.model_name}_best_checkpoint.pt')
    else:
        fpath = os.path.join(self.save_dir,
            f'{self.model_name}_epoch{epoch}_checkpoint.pt')
    print_once(f'Saving {fpath}...')
    torch.save(state, fpath)
    if not is_best:
        self.tracked[epoch] = fpath
        for epoch in (set(list(self.tracked)[:-2]) - set(self.keep_milestones)
            ):
            try:
                os.remove(self.tracked[epoch])
            except:
                pass
            del self.tracked[epoch]
