def get_inception_probs(inps):
    session = tf.get_default_session()
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    preds = np.zeros([inps.shape[0], 1000], dtype=np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] / 255.0 * 2 - 1
        preds[i * BATCH_SIZE:i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])
            ] = session.run(logits, {inception_images: inp})[:, :1000]
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    return preds
