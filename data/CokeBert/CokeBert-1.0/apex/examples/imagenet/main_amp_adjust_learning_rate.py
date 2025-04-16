def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30
    if epoch >= 80:
        factor = factor + 1
    lr = args.lr * 0.1 ** factor
    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5.0 * len_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
