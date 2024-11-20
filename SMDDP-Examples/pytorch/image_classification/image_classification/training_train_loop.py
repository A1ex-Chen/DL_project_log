def train_loop(model_and_loss, optimizer, scaler, lr_scheduler,
    train_loader, val_loader, logger, should_backup_checkpoint,
    steps_per_epoch, ema=None, model_ema=None, use_amp=False,
    batch_size_multiplier=1, best_prec1=0, start_epoch=0, end_epoch=0,
    early_stopping_patience=-1, prof=-1, skip_training=False,
    skip_validation=False, save_checkpoints=True, checkpoint_dir='./',
    checkpoint_filename='checkpoint.pth.tar'):
    prec1 = -1
    use_ema = model_ema is not None and ema is not None
    if early_stopping_patience > 0:
        epochs_since_improvement = 0
    backup_prefix = checkpoint_filename[:-len('checkpoint.pth.tar')
        ] if checkpoint_filename.endswith('checkpoint.pth.tar') else ''
    print(f'RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}')
    with utils.TimeoutHandler() as timeout_handler:
        interrupted = False
        for epoch in range(start_epoch, end_epoch):
            if logger is not None:
                logger.start_epoch()
            if not skip_training:
                interrupted = train(train_loader, model_and_loss, optimizer,
                    scaler, lr_scheduler, logger, epoch, steps_per_epoch,
                    timeout_handler, ema=ema, use_amp=use_amp, prof=prof,
                    register_metrics=epoch == start_epoch,
                    batch_size_multiplier=batch_size_multiplier)
            if not skip_validation:
                prec1, nimg = validate(val_loader, model_and_loss, logger,
                    epoch, use_amp=use_amp, prof=prof, register_metrics=
                    epoch == start_epoch)
                if use_ema:
                    model_ema.load_state_dict({k.replace('module.', ''): v for
                        k, v in ema.state_dict().items()})
                    prec1, nimg = validate(val_loader, model_ema, logger,
                        epoch, prof=prof, register_metrics=epoch ==
                        start_epoch, prefix='val_ema')
                if prec1 > best_prec1:
                    is_best = True
                    best_prec1 = prec1
                else:
                    is_best = False
            else:
                is_best = True
                best_prec1 = 0
            if logger is not None:
                logger.end_epoch()
            if save_checkpoints and (not dist.is_initialized() or dist.
                get_rank() == 0):
                if should_backup_checkpoint(epoch):
                    backup_filename = '{}checkpoint-{}.pth.tar'.format(
                        backup_prefix, epoch + 1)
                else:
                    backup_filename = None
                checkpoint_state = {'epoch': epoch + 1, 'state_dict':
                    model_and_loss.model.state_dict(), 'best_prec1':
                    best_prec1, 'optimizer': optimizer.state_dict()}
                if use_ema:
                    checkpoint_state['state_dict_ema'] = ema.state_dict()
                utils.save_checkpoint(checkpoint_state, is_best,
                    checkpoint_dir=checkpoint_dir, backup_filename=
                    backup_filename, filename=checkpoint_filename)
            if early_stopping_patience > 0:
                if not is_best:
                    epochs_since_improvement += 1
                else:
                    epochs_since_improvement = 0
                if epochs_since_improvement >= early_stopping_patience:
                    break
            if interrupted:
                break
