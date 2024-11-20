def lr_schedule(epoch):
    initial_learning_rate = 0.001
    decay_per_epoch = 0.99
    lrate = initial_learning_rate * decay_per_epoch ** epoch
    print('Learning rate = %f' % lrate)
    return lrate
