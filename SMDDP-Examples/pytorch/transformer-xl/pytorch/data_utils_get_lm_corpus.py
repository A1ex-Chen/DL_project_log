def get_lm_corpus(datadir, dataset, vocab):
    if vocab == 'word':
        fn = os.path.join(datadir, 'cache.pt')
    elif vocab == 'bpe':
        fn = os.path.join(datadir, 'cache.pt.bpe')
    else:
        raise RuntimeError('Unsupported vocab')
    if os.path.exists(fn):
        logging.info('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        logging.info('Producing dataset {}...'.format(dataset))
        kwargs = {}
        if dataset in ['wt103', 'wt2']:
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = False
        elif dataset == 'ptb':
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = True
        elif dataset == 'lm1b':
            kwargs['special'] = []
            kwargs['lower_case'] = False
            kwargs['vocab_file'] = os.path.join(datadir, '1b_word_vocab.txt')
        elif dataset in ['enwik8', 'text8']:
            pass
        corpus = Corpus(datadir, dataset, vocab, **kwargs)
        with utils.distributed.sync_workers() as rank:
            if rank == 0:
                torch.save(corpus, fn)
    return corpus
