def main(args, device):
    print('loading data.')
    vocab, c2i, i2c, _, _ = get_vocab_from_file(args.i + '/vocab.txt')
    model = CharRNN(len(vocab), len(vocab), max_len=args.maxlen).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    pt = torch.load(args.logdir + '/' + args.model, map_location=device)
    model.load_state_dict(pt['state_dict'])
    optimizer.load_state_dict(pt['optim_state_dict'])
    total_sampled = 0
    total_valid = 0
    total_unqiue = 0
    smiles = set()
    start = time.time()
    batch_size = args.batch_size
    for epoch in range(int(args.n / batch_size)):
        samples = sample(model, i2c, c2i, device, batch_size=batch_size,
            max_len=args.maxlen, temp=args.t)
        samples = list(map(lambda x: x[1:-1], samples))
        total_sampled += len(samples)
        if args.vb or args.vr:
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
    df.to_csv(args.o, index=False, header=True)
    print('output smiles to', args.o)
    print('Took ', end - start, 'seconds')
    print('Sampled', total_sampled)
    print('Total unique', total_unqiue, float(total_unqiue) / float(
        total_sampled))
    if args.vr or args.vb:
        print('total valid', total_valid, float(total_valid) / float(
            total_sampled))
