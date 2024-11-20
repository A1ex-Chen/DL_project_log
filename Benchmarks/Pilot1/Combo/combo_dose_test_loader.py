def test_loader(loader):
    x_train_list, y_train, x_val_list, y_val = loader.load_data()
    print('x_train shapes:')
    for x in x_train_list:
        print(x.shape)
    print('y_train shape:', y_train.shape)
    print('x_val shapes:')
    for x in x_val_list:
        print(x.shape)
    print('y_val shape:', y_val.shape)
