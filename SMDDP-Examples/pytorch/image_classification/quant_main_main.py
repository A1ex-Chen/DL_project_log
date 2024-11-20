def main(args, model_args, model_arch):
    exp_start_time = time.time()
    global best_prec1
    best_prec1 = 0
    skip_calibration = (args.skip_calibration or args.evaluate or args.
        resume is not None)
    select_default_calib_method()
    (model_and_loss, optimizer, lr_policy, scaler, train_loader, val_loader,
        logger, ema, model_ema, train_loader_len, batch_size_multiplier,
        start_epoch) = prepare_for_training(args, model_args, model_arch)
    print(f'RUNNING QUANTIZATION')
    if not skip_calibration:
        calibrate(model_and_loss.model, train_loader, logger, calib_iter=10)
    train_loop(model_and_loss, optimizer, scaler, lr_policy, train_loader,
        val_loader, logger, should_backup_checkpoint(args), ema=ema,
        model_ema=model_ema, steps_per_epoch=train_loader_len, use_amp=args
        .amp, batch_size_multiplier=batch_size_multiplier, start_epoch=
        start_epoch, end_epoch=min(start_epoch + args.run_epochs, args.
        epochs) if args.run_epochs != -1 else args.epochs, best_prec1=
        best_prec1, prof=args.prof, skip_training=args.evaluate,
        skip_validation=args.training_only, save_checkpoints=args.
        save_checkpoints, checkpoint_dir=args.workspace,
        checkpoint_filename='quantized_' + args.checkpoint_filename)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank(
        ) == 0:
        logger.end()
    print('Experiment ended')
