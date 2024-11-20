def __init__(self, args):
    with open('{}/preprocessed/twitter_q7_mapped.pickle'.format(args.
        root_dir), 'rb') as handler:
        data = pickle.load(handler)
    fews_shot_using_train_ratio = False
    if fews_shot_using_train_ratio:
        self.train, test = train_test_split(data, test_size=1 - args.
            fewshot_train_ratio, random_state=args.seed, shuffle=True)
        self.dev, self.test = train_test_split(test, test_size=0.8,
            random_state=args.seed, shuffle=True)
    else:
        random.seed(args.seed)
        random.shuffle(data)
        chunk_indx = args.fewshot_train
        if chunk_indx != None:
            self.train = data[:chunk_indx]
            self.dev, self.test = train_test_split(data[chunk_indx:],
                test_size=0.8, random_state=0, shuffle=True)
        else:
            train, self.test = train_test_split(data, test_size=0.1,
                random_state=0, shuffle=True)
            self.train, self.dev = train_test_split(train, test_size=0.15,
                random_state=0, shuffle=True)
