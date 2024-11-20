def given_indices(args, loader):
    test_gen = CombinedDataGenerator(loader, partition='test', batch_size=
        args.batch_size)
    index = read_IDs_file(args.uq_infer_file)
    test_gen.index = index
    test_gen.index_cycle = cycle(index)
    test_gen.size = len(index)
    test_gen.steps = np.ceil(test_gen.size / args.batch_size)
    return test_gen
