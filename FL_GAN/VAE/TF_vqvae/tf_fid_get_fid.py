def get_fid(images1, images2):
    session = tf.get_default_session()
    assert type(images1) == np.ndarray
    assert len(images1.shape) == 4
    assert type(images2) == np.ndarray
    assert len(images2.shape) == 4
    assert images1.shape == images2.shape, 'The two numpy arrays must have the same shape'
    print('Calculating FID with %i images from each distribution' % images1
        .shape[0])
    start_time = time.time()
    act1 = get_inception_activations(images1)
    act2 = get_inception_activations(images2)
    fid = activations2distance(act1, act2)
    print('FID calculation time: %f s' % (time.time() - start_time))
    return fid
