def given_drugs(args, loader):
    test_gen = CombinedDataGenerator(loader, partition='test', batch_size=
        args.batch_size)
    include_drugs = read_IDs_file(args.uq_infer_file)
    df_response = test_gen.data.df_response
    if np.isin('Drug', df_response.columns.values):
        df = df_response[['Drug']]
        index = df.index[df['Drug'].isin(include_drugs)]
    else:
        df = df_response[['Drug1', 'Drug2']]
        index = df.index[df['Drug1'].isin(include_drugs) | df['Drug2'].isin
            (include_drugs)]
    test_gen.index = index
    test_gen.index_cycle = cycle(index)
    test_gen.size = len(index)
    test_gen.steps = np.ceil(test_gen.size / args.batch_size)
    return test_gen
