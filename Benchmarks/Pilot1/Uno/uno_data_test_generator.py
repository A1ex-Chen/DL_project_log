def test_generator(loader):
    gen = CombinedDataGenerator(loader).flow()
    x_list, y = next(gen)
    print('x shapes:')
    for x in x_list:
        print(x.shape)
    print('y shape:')
    print(y.shape)
