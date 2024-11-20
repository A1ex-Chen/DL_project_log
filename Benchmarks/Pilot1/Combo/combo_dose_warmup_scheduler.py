def warmup_scheduler(epoch):
    lr = args.learning_rate or base_lr * args.batch_size / 100
    if epoch <= 5:
        K.set_value(model.optimizer.lr, (base_lr * (5 - epoch) + lr * epoch
            ) / 5)
    logger.debug('Epoch {}: lr={}'.format(epoch, K.get_value(model.
        optimizer.lr)))
    return K.get_value(model.optimizer.lr)
