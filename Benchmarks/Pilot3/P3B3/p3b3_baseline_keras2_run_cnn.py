def run_cnn(GP, train_x, train_y, test_x, test_y, learning_rate=0.01,
    batch_size=10, epochs=10, dropout=0.5, optimizer='adam', wv_len=300,
    filter_sizes=[3, 4, 5], num_filters=[300, 300, 300], emb_l2=0.001, w_l2
    =0.01):
    max_vocab = np.max(train_x)
    max_vocab2 = np.max(test_x)
    if max_vocab2 > max_vocab:
        max_vocab = max_vocab2
    wv_mat = np.random.randn(max_vocab + 1, wv_len).astype('float32') * 0.1
    task_list = GP['task_list']
    task_names = GP['task_names']
    num_classes = []
    for i in range(train_y.shape[1]):
        num_classes.append(np.max(train_y[:, i]) + 1)
    print('Num_classes = ', num_classes)
    kerasDefaults = candle.keras_default_config()
    optimizer_run = candle.build_optimizer(optimizer, learning_rate,
        kerasDefaults)
    cnn = keras_mt_shared_cnn.init_export_network(task_names=task_names,
        task_list=task_list, num_classes=num_classes, in_seq_len=1500,
        vocab_size=len(wv_mat), wv_space=wv_len, filter_sizes=filter_sizes,
        num_filters=num_filters, concat_dropout_prob=dropout, emb_l2=emb_l2,
        w_l2=w_l2, optimizer=optimizer_run)
    print(cnn.summary())
    val_labels = {}
    train_labels = []
    for i in range(train_y.shape[1]):
        if i in task_list:
            task_string = task_names[i]
            val_labels[task_string] = test_y[:, i]
            train_labels.append(np.array(train_y[:, i]))
    validation_data = {'Input': test_x}, val_labels
    candleRemoteMonitor = candle.CandleRemoteMonitor(params=GP)
    timeoutMonitor = candle.TerminateOnTimeOut(GP['timeout'])
    history = cnn.fit(x=np.array(train_x), y=train_labels, batch_size=
        batch_size, epochs=epochs, verbose=2, validation_data=
        validation_data, callbacks=[candleRemoteMonitor, timeoutMonitor])
    return history
