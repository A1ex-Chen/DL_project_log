def step_decay(epoch):
    global initial_lrate, epochs_drop, drop
    lrate = initial_lrate * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    return lrate
