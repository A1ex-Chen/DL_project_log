def run():
    args = initialize_parameters()
    get_custom_objects()['PermanentDropout'] = PermanentDropout
    model = keras.models.load_model(args.model_file, compile=False)
    model.load_weights(args.weights_file)
    df_expr, df_desc = prepare_data(sample_set=args.sample_set, drug_set=
        args.drug_set, use_landmark_genes=args.use_landmark_genes)
    print('total available samples: ', df_expr[['Sample']].shape[0])
    print('total available drugs: ', df_desc[['Drug']].shape[0])
    if args.ns > 0 and args.si > 0:
        df_sample_ids = df_expr[['Sample']].iloc[args.si:args.si + args.ns]
    elif args.si > 0:
        df_sample_ids = df_expr[['Sample']].iloc[args.si:]
    elif args.ns > 0:
        df_sample_ids = df_expr[['Sample']].head(args.ns)
    else:
        df_sample_ids = df_expr[['Sample']].copy()
    if args.nd > 0:
        df_drug_ids = df_desc[['Drug']].head(args.nd)
    else:
        df_drug_ids = df_desc[['Drug']].copy()
    df_sum = cross_join3(df_sample_ids, df_drug_ids, df_drug_ids, suffixes=
        ('1', '2'))
    n_samples = df_sample_ids.shape[0]
    n_drugs = df_drug_ids.shape[0]
    n_rows = n_samples * n_drugs * n_drugs
    print(
        'Predicting drug response for {} combinations: {} samples x {} drugs x {} drugs'
        .format(n_rows, n_samples, n_drugs, n_drugs))
    n = args.n_pred
    df_sum['N'] = n
    df_seq = pd.DataFrame({'Seq': range(1, n + 1)})
    df_all = cross_join(df_sum, df_seq)
    total = df_sum.shape[0]
    for i in tqdm(range(0, total, args.step)):
        j = min(i + args.step, total)
        x_all_list = []
        df_x_all = pd.merge(df_all[['Sample']].iloc[i:j], df_expr, on=
            'Sample', how='left')
        x_all_list.append(df_x_all.drop(['Sample'], axis=1).values)
        drugs = ['Drug1', 'Drug2']
        for drug in drugs:
            df_x_all = pd.merge(df_all[[drug]].iloc[i:j], df_desc, left_on=
                drug, right_on='Drug', how='left')
            x_all_list.append(df_x_all.drop([drug, 'Drug'], axis=1).values)
        preds = []
        for k in range(n):
            y_pred = model.predict(x_all_list, batch_size=args.batch_size,
                verbose=0).flatten()
            preds.append(y_pred)
            df_all.loc[i * n + k:(j - 1) * n + k:n, 'PredGrowth'] = y_pred
            df_all.loc[i * n + k:(j - 1) * n + k:n, 'Seq'] = k + 1
        if n > 0:
            df_sum.loc[i:j - 1, 'PredGrowthMean'] = np.mean(preds, axis=0)
            df_sum.loc[i:j - 1, 'PredGrowthStd'] = np.std(preds, axis=0)
            df_sum.loc[i:j - 1, 'PredGrowthMin'] = np.min(preds, axis=0)
            df_sum.loc[i:j - 1, 'PredGrowthMax'] = np.max(preds, axis=0)
    csv_all = 'comb_pred_{}_{}.all.tsv'.format(args.sample_set, args.drug_set)
    df_all.to_csv(csv_all, index=False, sep='\t', float_format='%.4f')
    if n > 0:
        csv = 'comb_pred_{}_{}.tsv'.format(args.sample_set, args.drug_set)
        df_sum.to_csv(csv, index=False, sep='\t', float_format='%.4f')
