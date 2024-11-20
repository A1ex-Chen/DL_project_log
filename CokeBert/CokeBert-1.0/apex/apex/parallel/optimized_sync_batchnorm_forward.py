def forward(self, input):
    channel_last = self.channel_last if input.dim() != 2 else True
    if not self.training and self.track_running_stats and not channel_last:
        return F.batch_norm(input, self.running_mean, self.running_var,
            self.weight, self.bias, False, 0.0, self.eps)
    else:
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.
                    num_batches_tracked)
            else:
                exponential_average_factor = self.momentum
        return SyncBatchnormFunction.apply(input, self.weight, self.bias,
            self.running_mean, self.running_var, self.eps, self.training or
            not self.track_running_stats, exponential_average_factor, self.
            process_group, channel_last)
