def activations2distance(act1, act2):
    return session.run(fcd, feed_dict={activations1: act1, activations2: act2})
