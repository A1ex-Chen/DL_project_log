def given_cells(args, loader):
    test_gen = CombinedDataGenerator(loader, partition='test', batch_size=
        args.batch_size)
    include_cells = read_IDs_file(args.uq_infer_file)
    df = test_gen.data.df_response[['Sample']]
    index = df.index[df['Sample'].isin(include_cells)]
    test_gen.index = index
    test_gen.index_cycle = cycle(index)
    test_gen.size = len(index)
    test_gen.steps = np.ceil(test_gen.size / args.batch_size)
    return test_gen
