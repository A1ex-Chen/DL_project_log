def save_and_test_saved_model(params, model, root_fname, nb_classes, alpha,
    mask, X_train, X_test, Y_test):
    model_json = model.to_json()
    with open(params['save_path'] + root_fname + '.model.json', 'w'
        ) as json_file:
        json_file.write(model_json)
    model.save_weights(params['save_path'] + root_fname + '.model.h5')
    print('Saved model to disk')
    json_file = open(params['save_path'] + root_fname + '.model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)
    loaded_model_json.load_weights(params['save_path'] + root_fname +
        '.model.h5')
    loaded_model_json.compile(loss=candle.abstention_loss(alpha, mask),
        optimizer='SGD', metrics=[candle.abstention_acc_metric(nb_classes)])
    score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)
    print('json Validation abstention loss:', score_json[0])
    print('json Validation abstention accuracy:', score_json[1])
    print('json %s: %.2f%%' % (loaded_model_json.metrics_names[1], 
        score_json[1] * 100))
    predict_train = loaded_model_json.predict(X_train)
    predict_test = loaded_model_json.predict(X_test)
    print('train_shape:', predict_train.shape)
    print('test_shape:', predict_test.shape)
    predict_train_classes = np.argmax(predict_train, axis=1)
    predict_test_classes = np.argmax(predict_test, axis=1)
    np.savetxt(params['save_path'] + root_fname + '_predict_train.csv',
        predict_train, delimiter=',', fmt='%.3f')
    np.savetxt(params['save_path'] + root_fname + '_predict_test.csv',
        predict_test, delimiter=',', fmt='%.3f')
    np.savetxt(params['save_path'] + root_fname +
        '_predict_train_classes.csv', predict_train_classes, delimiter=',',
        fmt='%d')
    np.savetxt(params['save_path'] + root_fname +
        '_predict_test_classes.csv', predict_test_classes, delimiter=',',
        fmt='%d')
