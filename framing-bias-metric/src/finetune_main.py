def main(args, model=None) ->SummarizationModule:
    Path(args.output_dir).mkdir(exist_ok=True)
    check_output_dir(args, expected_items=3)
    args.default_root_dir = args.output_dir
    if model is None:
        model: SummarizationModule = SummarizationModule(args)
    dataset = Path(args.data_dir).name
    if args.logger_name == 'default' or args.fast_dev_run or str(args.
        output_dir).startswith('/tmp') or str(args.output_dir).startswith(
        '/var'):
        logger = True
    elif args.logger_name == 'wandb':
        from pytorch_lightning.loggers import WandbLogger
        project = os.environ.get('WANDB_PROJECT', dataset)
        logger = WandbLogger(name=model.output_dir.name, project=project)
    elif args.logger_name == 'wandb_shared':
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(name=model.output_dir.name, project=
            f'hf_{dataset}')
    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.
            early_stopping_patience)
    else:
        es_callback = False
    lower_is_better = args.val_metric == 'loss'
    trainer: pl.Trainer = generic_train(model, args, logging_callback=
        Seq2SeqLoggingCallback(), checkpoint_callback=
        get_checkpoint_callback(args.output_dir, model.val_metric, args.
        save_top_k, lower_is_better), early_stopping_callback=es_callback,
        logger=logger)
    pickle_save(model.hparams, model.output_dir / 'hparams.pkl')
    if not args.do_predict:
        return model
    model.hparams.test_checkpoint = ''
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir,
        '*.ckpt'), recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)
    if checkpoints:
        print('Loading checkpoints from: ', model.hparams.test_checkpoint)
        ckpt = pl_load(model.hparams.test_checkpoint, map_location=lambda
            storage, loc: storage)
        model.load_state_dict(ckpt['state_dict'])
        trainer.test(model)
    else:
        print('No checkpoint exists! Using pretrained models')
        trainer.test(model)
    return model
