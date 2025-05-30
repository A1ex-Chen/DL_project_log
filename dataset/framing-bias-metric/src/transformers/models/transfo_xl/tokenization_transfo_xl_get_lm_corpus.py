@torch_only_method
def get_lm_corpus(datadir, dataset):
    fn = os.path.join(datadir, 'cache.pt')
    fn_pickle = os.path.join(datadir, 'cache.pkl')
    if os.path.exists(fn):
        logger.info('Loading cached dataset...')
        corpus = torch.load(fn_pickle)
    elif os.path.exists(fn):
        logger.info('Loading cached dataset from pickle...')
        with open(fn, 'rb') as fp:
            corpus = pickle.load(fp)
    else:
        logger.info('Producing dataset {}...'.format(dataset))
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
        corpus = TransfoXLCorpus(datadir, dataset, **kwargs)
        torch.save(corpus, fn)
    return corpus
