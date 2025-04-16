def smart_resume(ckpt, optimizer, ema=None, weights='yolov5s.pt', epochs=
    300, resume=True):
    best_fitness = 0.0
    start_epoch = ckpt['epoch'] + 1
    if ckpt['optimizer'] is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
        best_fitness = ckpt['best_fitness']
    if ema and ckpt.get('ema'):
        ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
        ema.updates = ckpt['updates']
    if resume:
        assert start_epoch > 0, f"""{weights} training to {epochs} epochs is finished, nothing to resume.
Start a new training without --resume, i.e. 'python train.py --weights {weights}'"""
        LOGGER.info(
            f'Resuming training from {weights} from epoch {start_epoch} to {epochs} total epochs'
            )
    if epochs < start_epoch:
        LOGGER.info(
            f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs."
            )
        epochs += ckpt['epoch']
    return best_fitness, start_epoch, epochs
