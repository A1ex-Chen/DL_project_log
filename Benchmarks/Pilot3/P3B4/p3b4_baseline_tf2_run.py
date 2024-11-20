def run(gParameters):
    fpath = fetch_data(gParameters)
    learning_rate = gParameters['learning_rate']
    batch_size = gParameters['batch_size']
    epochs = gParameters['epochs']
    dropout = gParameters['dropout']
    embed_train = gParameters['embed_train']
    optimizer = gParameters['optimizer']
    wv_len = gParameters['wv_len']
    attention_size = gParameters['attention_size']
    attention_heads = gParameters['attention_heads']
    max_words = gParameters['max_words']
    max_lines = gParameters['max_lines']
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
    train_x = train_x.reshape((train_x.shape[0], max_lines, max_words))
    test_x = test_x.reshape((test_x.shape[0], max_lines, max_words))
    X_train = train_x
    X_test = test_x
    y_trains = []
    y_tests = []
    for k in range(num_tasks):
        y_trains.append(train_y[:, k])
        y_tests.append(test_y[:, k])
    model = mthisan(vocab, num_classes, max_lines, max_words,
        attention_heads=attention_heads, attention_size=attention_size)
    ret = model.train(X_train, y_trains, batch_size=batch_size, epochs=
        epochs, validation_data=[X_test, y_tests])
    return ret
