def main(args, model_args, model_arch):
    global best_prec1
    best_prec1 = 0
    (model_and_loss, optimizer, lr_policy, scaler, train_loader, val_loader,
        logger, ema, model_ema, train_loader_len, batch_size_multiplier,
        start_epoch) = prepare_for_training(args, model_args, model_arch)
    exp_start_time = time.time()
    train_loop(model_and_loss, optimizer, scaler, lr_policy, train_loader,
        val_loader, logger, should_backup_checkpoint(args), ema=ema,
        model_ema=model_ema, steps_per_epoch=train_loader_len, use_amp=args
        .amp, batch_size_multiplier=batch_size_multiplier, start_epoch=
        start_epoch, end_epoch=min(start_epoch + args.run_epochs, args.
        epochs) if args.run_epochs != -1 else args.epochs,
        early_stopping_patience=args.early_stopping_patience, best_prec1=
        best_prec1, prof=args.prof, skip_training=args.evaluate,
        skip_validation=args.training_only, save_checkpoints=args.
        save_checkpoints and not args.evaluate, checkpoint_dir=args.
        workspace, checkpoint_filename=args.checkpoint_filename)
    exp_duration = time.time() - exp_start_time
    if not dist.is_initialized() or dist.get_rank() == 0:
        print('Experiment ended')
        print('Total training time: {:.2f} secs'.format(exp_duration))
        logger.end()
