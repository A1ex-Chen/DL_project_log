def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
    return min(1, (epoch * num_iters_per_epoch + iters) / (2 *
        num_iters_per_epoch))
