def step_function(epoch, lr):
    if epoch < 12:
        return 0.0005
    elif epoch < 24:
        return 0.0001
    elif epoch < 36:
        return 2e-05
    else:
        return 1e-05
