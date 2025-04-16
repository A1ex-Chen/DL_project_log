def run(params):
    args = candle.ArgumentStruct(**params)
    candle.set_seed(args.rng_seed)
    ext = extension_from_parameters(args)
    verify_path(args.save_path)
    prefix = args.save_path + ext
    logfile = args.logfile if args.logfile else prefix + '.log'
    set_up_logger(logfile, args.verbose)
    logger.info('Params: {}'.format(params))
    loader = ComboDataLoader(seed=args.rng_seed, val_split=args.val_split,
        cell_features=args.cell_features, drug_features=args.drug_features,
        use_mean_growth=args.use_mean_growth, response_url=args.
        response_url, use_landmark_genes=args.use_landmark_genes,
        preprocess_rnaseq=args.preprocess_rnaseq, exclude_cells=args.
        exclude_cells, exclude_drugs=args.exclude_drugs, use_combo_score=
        args.use_combo_score, cv_partition=args.cv_partition, cv=args.cv)
    train_gen = ComboDataGenerator(loader, batch_size=args.batch_size).flow()
    val_gen = ComboDataGenerator(loader, partition='val', batch_size=args.
        batch_size).flow()
    train_steps = int(loader.n_train / args.batch_size)
    val_steps = int(loader.n_val / args.batch_size)
    model = build_model(loader, args, verbose=True)
    model.summary()
    if args.cp:
        model_json = model.to_json()
        with open(prefix + '.model.json', 'w') as f:
            print(model_json, file=f)

    def warmup_scheduler(epoch):
        lr = args.learning_rate or base_lr * args.batch_size / 100
        if epoch <= 5:
            K.set_value(model.optimizer.lr, (base_lr * (5 - epoch) + lr *
                epoch) / 5)
        logger.debug('Epoch {}: lr={}'.format(epoch, K.get_value(model.
            optimizer.lr)))
        return K.get_value(model.optimizer.lr)
    df_pred_list = []
    cv_ext = ''
    cv = args.cv if args.cv > 1 else 1
    fold = 0
    while fold < cv:
        if args.cv > 1:
            logger.info('Cross validation fold {}/{}:'.format(fold + 1, cv))
            cv_ext = '.cv{}'.format(fold + 1)
        model = build_model(loader, args)
        optimizer = optimizers.deserialize({'class_name': args.optimizer,
            'config': {}})
        base_lr = args.base_lr or K.get_value(optimizer.lr)
        if args.learning_rate:
            K.set_value(optimizer.lr, args.learning_rate)
        model.compile(loss=args.loss, optimizer=optimizer, metrics=[mae, r2])
        params.update(candle.compute_trainable_params(model))
        candle_monitor = candle.CandleRemoteMonitor(params=params)
        timeout_monitor = candle.TerminateOnTimeOut(params['timeout'])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-05)
        warmup_lr = LearningRateScheduler(warmup_scheduler)
        checkpointer = ModelCheckpoint(prefix + cv_ext + '.weights.h5',
            save_best_only=True, save_weights_only=True)
        tensorboard = TensorBoard(log_dir='tb/tb{}{}'.format(ext, cv_ext))
        history_logger = LoggingCallback(logger.debug)
        model_recorder = ModelRecorder()
        callbacks = [candle_monitor, timeout_monitor, history_logger,
            model_recorder]
        if args.reduce_lr:
            callbacks.append(reduce_lr)
        if args.warmup_lr:
            callbacks.append(warmup_lr)
        if args.cp:
            callbacks.append(checkpointer)
        if args.tb:
            callbacks.append(tensorboard)
        if args.gen:
            history = model.fit_generator(train_gen, train_steps, epochs=
                args.epochs, callbacks=callbacks, validation_data=val_gen,
                validation_steps=val_steps)
            fold += 1
        else:
            if args.cv > 1:
                (x_train_list, y_train, x_val_list, y_val, df_train, df_val
                    ) = loader.load_data_cv(fold)
            else:
                (x_train_list, y_train, x_val_list, y_val, df_train, df_val
                    ) = loader.load_data()
            y_shuf = np.random.permutation(y_val)
            log_evaluation(evaluate_prediction(y_val, y_shuf), description=
                'Between random pairs in y_val:')
            history = model.fit(x_train_list, y_train, batch_size=args.
                batch_size, shuffle=args.shuffle, epochs=args.epochs,
                callbacks=callbacks, validation_data=(x_val_list, y_val))
        if args.cp:
            model.load_weights(prefix + cv_ext + '.weights.h5')
        if not args.gen:
            y_val_pred = model.predict(x_val_list, batch_size=args.batch_size
                ).flatten()
            scores = evaluate_prediction(y_val, y_val_pred)
            if args.cv > 1 and scores[args.loss] > args.max_val_loss:
                logger.warn(
                    'Best val_loss {} is greater than {}; retrain the model...'
                    .format(scores[args.loss], args.max_val_loss))
                continue
            else:
                fold += 1
            log_evaluation(scores)
            df_val.is_copy = False
            df_val.loc[:, 'GROWTH_PRED'] = y_val_pred
            df_val.loc[:, 'GROWTH_ERROR'] = y_val_pred - y_val
            df_pred_list.append(df_val)
        if args.cp:
            model_recorder.best_model.save(prefix + '.model.h5')
        candle.plot_history(prefix, history, 'loss')
        candle.plot_history(prefix, history, 'r2')
        if K.backend() == 'tensorflow':
            K.clear_session()
    if not args.gen:
        if args.use_combo_score:
            pred_fname = prefix + '.predicted.score.tsv'
        elif args.use_mean_growth:
            pred_fname = prefix + '.predicted.mean.growth.tsv'
        else:
            pred_fname = prefix + '.predicted.growth.tsv'
        df_pred = pd.concat(df_pred_list)
        df_pred.to_csv(pred_fname, sep='\t', index=False, float_format='%.4g')
    logger.handlers = []
    return history
