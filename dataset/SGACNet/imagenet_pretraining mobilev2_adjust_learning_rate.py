def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * 0.1 ** (epoch // 30)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
