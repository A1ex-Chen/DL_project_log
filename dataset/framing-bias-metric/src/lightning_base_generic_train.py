def generic_train(model: BaseTransformer, args: argparse.Namespace,
    early_stopping_callback=None, logger=True, extra_callbacks=[],
    checkpoint_callback=None, logging_callback=None, **extra_train_kwargs):
    pl.seed_everything(args.seed)
    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=args.
            output_dir, prefix='checkpoint', monitor='val_loss', mode='min',
            save_top_k=1)
    if early_stopping_callback:
        extra_callbacks.append(early_stopping_callback)
    if logging_callback is None:
        logging_callback = LoggingCallback()
    train_params = {}
    if args.fp16:
        train_params['precision'] = 16
        train_params['amp_level'] = args.fp16_opt_level
    if args.gpus > 1:
        train_params['distributed_backend'] = 'ddp'
    train_params['accumulate_grad_batches'] = args.accumulate_grad_batches
    train_params['accelerator'] = extra_train_kwargs.get('accelerator', None)
    train_params['profiler'] = extra_train_kwargs.get('profiler', None)
    trainer = pl.Trainer.from_argparse_args(args, weights_summary=None,
        callbacks=[logging_callback] + extra_callbacks, logger=logger,
        checkpoint_callback=checkpoint_callback, **train_params)
    if args.do_train:
        trainer.fit(model)
    return trainer
