def main(args):
    if args.permute_smiles != 0:
        try:
            randomSmiles('CNOPc1ccccc1', 10)
        except Exception:
            print(
                'Must set --permute_smiles to 0, cannot import RdKit. Smiles validity not being checked either.'
                )
    if args.start:
        count = 0
        vocab = set()
        vocab.update([START_CHAR, END_CHAR])
        print(vocab)
        with open(args.i, 'r') as f:
            for line in f:
                smi = line.strip()
                count += 1
                if len(smi) > args.maxlen - 2:
                    continue
                vocab.update(smi)
        vocab = list(vocab)
        with open(args.o + '/vocab.txt', 'w') as f:
            for v in vocab:
                f.write(v + '\n')
        print('Read ', count, 'smiles.')
        print('Vocab length: ', len(vocab), 'Max len: ', args.maxlen)
    count = 0
    _, c2i, _, _, _ = get_vocab_from_file(args.o + '/vocab.txt')
    count = 0
    with open(args.i, 'r') as f:
        with open(args.o + '/out.txt', 'w') as o:
            with multiprocessing.Pool(args.p) as p:
                smiss = p.imap(partial(randomSmiles, max_len=args.maxlen,
                    attempts=args.permute_smiles), map(lambda x: x.strip(), f))
                for smis in tqdm(smiss):
                    if smis is None:
                        continue
                    for smi in smis:
                        if len(smi) > args.maxlen - 2:
                            continue
                        try:
                            i = list(map(lambda x: str(c2i(x)), smi))
                            if i is not None:
                                o.write(','.join(i) + '\n')
                                count += 1
                        except Exception:
                            print('key error did not print.', count)
                            continue
    print('Output', count, 'smiles.')
