def run(params):
    print(
        'Note: This script is very picky. Please check device output to see where this is running. '
        )
    args = candle.ArgumentStruct(**params)
    data_url = args.data_url
    if args.model == 'ft_goodperforming_model.pt':
        file = 'pilot1/ft_goodperforming_model.pt'
    elif args.model == 'ft_poorperforming_model.pt':
        file = 'pilot1/ft_poorperforming_model.pt'
    else:
        file = 'mosesrun/autosave.model.pt'
    print('Recovering trained model')
    trained = candle.fetch_file(data_url + file, subdir='examples/rnngen')
    if args.use_gpus and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Using device:', device)
    print('loading data.')
    vocab, c2i, i2c, _, _ = get_vocab_from_file(args.input + '/vocab.txt')
    model = CharRNN(len(vocab), len(vocab), max_len=args.maxlen).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print('Loading trained model.')
    pt = torch.load(trained, map_location=device)
    model.load_state_dict(pt['state_dict'])
    optimizer.load_state_dict(pt['optim_state_dict'])
    print('Applying to loaded data')
    total_sampled = 0
    total_valid = 0
    total_unqiue = 0
    smiles = set()
    start = time.time()
    batch_size = args.batch_size
    for epoch in range(int(args.nsamples / batch_size)):
        samples = sample(model, i2c, c2i, device, batch_size=batch_size,
            max_len=args.maxlen, temp=args.temperature)
        samples = list(map(lambda x: x[1:-1], samples))
        total_sampled += len(samples)
        if args.vr or args.vb:
            valid_smiles, goods = count_valid_samples(samples, rdkit=args.vr)
            total_valid += valid_smiles
            smiles.update(goods)
        else:
            smiles.update(samples)
    smiles = list(smiles)
    total_unqiue += len(smiles)
    end = time.time()
    df = pd.DataFrame()
    df['smiles'] = smiles
    df.to_csv(args.output, index=False, header=True)
    print('output smiles to', args.output)
    print('Took ', end - start, 'seconds')
    print('Sampled', total_sampled)
    print('Total unique', total_unqiue, float(total_unqiue) / float(
        total_sampled))
    if args.vr or args.vb:
        print('total valid', total_valid, float(total_valid) / float(
            total_sampled))
