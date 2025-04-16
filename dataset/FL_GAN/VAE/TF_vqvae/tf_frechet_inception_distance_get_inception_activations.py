def get_inception_activations(inps):
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    act = np.zeros([inps.shape[0], 2048], dtype=np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] / 255.0 * 2 - 1
        act[i * BATCH_SIZE:i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])
            ] = session.run(activations, feed_dict={inception_images: inp})
    return act
