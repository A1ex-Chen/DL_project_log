def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch,
    boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [(initial_learning_rate * decay) for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [(itr < b) for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]
    return learning_rate_fn
