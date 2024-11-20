def initialize_parameters(default_model='p2b1_default_model.txt'):
    p2b1Bmk = p2b1.BenchmarkP2B1(p2b1.file_path, default_model, 'keras',
        prog='p2b1_baseline', desc=
        'Train Molecular Frame Autoencoder - Pilot 2 Benchmark 1')
    GP = candle.finalize_parameters(p2b1Bmk)
    print('\nTraining parameters:')
    for key in sorted(GP):
        print('\t%s: %s' % (key, GP[key]))
    if GP['backend'] != 'theano' and GP['backend'] != 'tensorflow':
        sys.exit('Invalid backend selected: %s' % GP['backend'])
    os.environ['KERAS_BACKEND'] = GP['backend']
    reload(K)
    """
    if GP['backend'] == 'theano':
        K.set_image_dim_ordering('th')
    elif GP['backend'] == 'tensorflow':
        K.set_image_dim_ordering('tf')
    """
    K.set_image_data_format('channels_last')
    print('Image data format: ', K.image_data_format())
    return GP
