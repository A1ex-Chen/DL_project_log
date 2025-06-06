def run(gParameters):
    fpath = fetch_data(gParameters)
    kerasDefaults = candle.keras_default_config()
    learning_rate = gParameters['learning_rate']
    batch_size = gParameters['batch_size']
    epochs = gParameters['epochs']
    dropout = gParameters['dropout']
    embed_train = gParameters['embed_train']
    optimizer = gParameters['optimizer']
    wv_len = gParameters['wv_len']
    attention_size = gParameters['attention_size']
    max_words = gParameters['max_words']
    max_lines = gParameters['max_lines']
    min_words = gParameters['min_words']
    min_lines = gParameters['min_lines']
    train_x = np.load(fpath + '/train_X.npy')
    train_y = np.load(fpath + '/train_Y.npy')
    test_x = np.load(fpath + '/test_X.npy')
    test_y = np.load(fpath + '/test_Y.npy')
    num_classes = []
    for task in range(len(train_y[0, :])):
        cat = np.unique(train_y[:, task])
        num_classes.append(len(cat))
        train_y[:, task] = [np.where(cat == x)[0][0] for x in train_y[:, task]]
        test_y[:, task] = [np.where(cat == x)[0][0] for x in test_y[:, task]]
    num_tasks = len(num_classes)
    max_vocab = np.max(train_x)
    max_vocab2 = np.max(test_x)
    if max_vocab2 > max_vocab:
        max_vocab = max_vocab2
    vocab_size = max_vocab + 1
    vocab = np.random.rand(vocab_size, wv_len)
    train_samples = train_x.shape[0]
    test_samples = test_x.shape[0]
    train_x = train_x.reshape((train_x.shape[0], max_lines, max_words))
    test_x = test_x.reshape((test_x.shape[0], max_lines, max_words))
    mask = []
    for i in range(train_samples + test_samples):
        doc_mask = np.ones((1, max_lines, max_words))
        num_lines = np.random.randint(min_lines, max_lines)
        for j in range(num_lines):
            num_words = np.random.randint(min_words, max_words)
            doc_mask[0, j, :num_words] = 0
        mask.append(doc_mask)
    mask = np.concatenate(mask, 0)
    model = hcan(vocab, num_classes, max_lines, max_words, attention_size=
        attention_size, dropout_rate=dropout, lr=learning_rate, optimizer=
        optimizer, embed_train=embed_train)
    ret = model.train(train_x, [np.array(train_y[:, 0]), np.array(train_y[:,
        1]), np.array(train_y[:, 2]), np.array(train_y[:, 3])], batch_size=
        batch_size, epochs=epochs, validation_data=[test_x, [np.array(
        test_y[:, 0]), np.array(test_y[:, 1]), np.array(test_y[:, 2]), np.
        array(test_y[:, 3])]])
    return ret
