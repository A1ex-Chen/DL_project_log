def run(params):
    args = candle.ArgumentStruct(**params)
    candle.set_seed(args.rng_seed)
    logfile_def = 'uno_infer_from_' + args.uq_infer_file + '.log'
    logfile = args.logfile if args.logfile else logfile_def
    candle.set_up_logger(logfile, logger, args.verbose)
    logger.info('Params: {}'.format(params))
    ext = extension_from_parameters(args)
    candle.verify_path(args.save_path)
    prefix = args.save_path + 'uno' + ext
    candle.register_permanent_dropout()
    model = keras.models.load_model(args.model_file, compile=False)
    model.load_weights(args.weights_file)
    logger.info('Loaded model:')
    model.summary(print_fn=logger.info)
    target = args.agg_dose or 'Growth'
    if (args.uq_infer_given_drugs or args.uq_infer_given_cells or args.
        uq_infer_given_indices):
        loader = CombinedDataLoader(args.rng_seed)
        loader.load(cache=args.cache, ncols=args.feature_subsample,
            agg_dose=args.agg_dose, cell_features=args.cell_features,
            drug_features=args.drug_features, drug_median_response_min=args
            .drug_median_response_min, drug_median_response_max=args.
            drug_median_response_max, use_landmark_genes=args.
            use_landmark_genes, use_filtered_genes=args.use_filtered_genes,
            cell_feature_subset_path=args.cell_feature_subset_path or args.
            feature_subset_path, drug_feature_subset_path=args.
            drug_feature_subset_path or args.feature_subset_path,
            preprocess_rnaseq=args.preprocess_rnaseq, single=args.single,
            train_sources=args.train_sources, test_sources=args.
            test_sources, embed_feature_source=not args.no_feature_source,
            encode_response_source=not args.no_response_source)
        if args.uq_infer_given_drugs:
            test_gen = given_drugs(args, loader)
        elif args.uq_infer_given_cells:
            test_gen = given_cells(args, loader)
        else:
            test_gen = given_indices(args, loader)
    else:
        test_gen = from_file(args, model)
    df_test = test_gen.get_response(copy=True)
    y_test = df_test[target].values
    for i in range(args.n_pred):
        if args.no_gen:
            x_test_list, y_test = test_gen.get_slice(size=test_gen.size,
                single=args.single)
            y_test_pred = model.predict(x_test_list, batch_size=args.batch_size
                )
        else:
            test_gen.reset()
            y_test_pred = model.predict(test_gen.flow(single=args.single),
                steps=test_gen.steps)
            y_test_pred = y_test_pred[:test_gen.size]
        if args.loss == 'het':
            y_test_pred_ = y_test_pred[:, 0]
            s_test_pred = y_test_pred[:, 1]
            y_test_pred = y_test_pred_.flatten()
            df_test['Predicted_' + target + '_' + str(i + 1)] = y_test_pred
            df_test['Pred_S_' + target + '_' + str(i + 1)] = s_test_pred
            pred_fname = prefix + '.predicted_INFER_HET.tsv'
        elif args.loss == 'qtl':
            y_test_pred_50q = y_test_pred[:, 0]
            y_test_pred_10q = y_test_pred[:, 1]
            y_test_pred_90q = y_test_pred[:, 2]
            y_test_pred = y_test_pred_50q.flatten()
            df_test['Predicted_50q_' + target + '_' + str(i + 1)] = y_test_pred
            df_test['Predicted_10q_' + target + '_' + str(i + 1)
                ] = y_test_pred_10q.flatten()
            df_test['Predicted_90q_' + target + '_' + str(i + 1)
                ] = y_test_pred_90q.flatten()
            pred_fname = prefix + '.predicted_INFER_QTL.tsv'
        else:
            y_test_pred = y_test_pred.flatten()
            df_test['Predicted_' + target + '_' + str(i + 1)] = y_test_pred
            pred_fname = prefix + '.predicted_INFER.tsv'
        if args.n_pred < 21:
            scores = evaluate_prediction(y_test, y_test_pred)
            log_evaluation(scores, logger)
    df_pred = df_test
    if args.agg_dose:
        if args.single:
            df_pred.sort_values(['Sample', 'Drug1', target], inplace=True)
        else:
            df_pred.sort_values(['Sample', 'Drug1', 'Drug2', target],
                inplace=True)
    elif args.single:
        df_pred.sort_values(['Sample', 'Drug1', 'Dose1', 'Growth'], inplace
            =True)
    else:
        df_pred.sort_values(['Sample', 'Drug1', 'Drug2', 'Dose1', 'Dose2',
            'Growth'], inplace=True)
    df_pred.to_csv(pred_fname, sep='\t', index=False, float_format='%.4g')
    logger.info('Predictions stored in file: {}'.format(pred_fname))
    if K.backend() == 'tensorflow':
        K.clear_session()
    logger.handlers = []
