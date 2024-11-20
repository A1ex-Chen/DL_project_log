def get_inception_score(images, splits=10):
    assert type(images) == np.ndarray
    assert len(images.shape) == 4
    assert images.shape[1] == 3
    assert np.min(images[0]) >= 0 and np.max(images[0]
        ) > 10, 'Image values should be in the range [0, 255]'
    print('Calculating Inception Score with %i images in %i splits' % (
        images.shape[0], splits))
    start_time = time.time()
    preds = get_inception_probs(images)
    mean, std = preds2score(preds, splits)
    print('Inception Score calculation time: %f s' % (time.time() - start_time)
        )
    return mean, std
