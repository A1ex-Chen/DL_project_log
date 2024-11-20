def forward(self, input):
    torch.cuda.nvtx.range_push('sync_bn_fw_with_mean_var')
    mean = None
    var = None
    cast = None
    out = None
    if self.running_mean is not None:
        if self.running_mean.dtype != input.dtype:
            input = input.to(self.running_mean.dtype)
            cast = input.dtype
    elif self.weight is not None:
        if self.weight.dtype != input.dtype:
            input = input.to(self.weight.dtype)
            cast = input.dtype
    if not self.training and self.track_running_stats:
        torch.cuda.nvtx.range_pop()
        out = F.batch_norm(input, self.running_mean, self.running_var, self
            .weight, self.bias, False, 0.0, self.eps)
    else:
        process_group = self.process_group
        world_size = 1
        if not self.process_group:
            process_group = torch.distributed.group.WORLD
        self.num_batches_tracked += 1
        with torch.no_grad():
            channel_first_input = input.transpose(0, 1).contiguous()
            squashed_input_tensor_view = channel_first_input.view(
                channel_first_input.size(0), -1)
            m = None
            local_m = float(squashed_input_tensor_view.size()[1])
            local_mean = torch.mean(squashed_input_tensor_view, 1)
            local_sqr_mean = torch.pow(squashed_input_tensor_view, 2).mean(1)
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size(process_group)
                torch.distributed.all_reduce(local_mean, ReduceOp.SUM,
                    process_group)
                mean = local_mean / world_size
                torch.distributed.all_reduce(local_sqr_mean, ReduceOp.SUM,
                    process_group)
                sqr_mean = local_sqr_mean / world_size
                m = local_m * world_size
            else:
                m = local_m
                mean = local_mean
                sqr_mean = local_sqr_mean
            var = sqr_mean - mean.pow(2)
            if self.running_mean is not None:
                self.running_mean = self.momentum * mean + (1 - self.momentum
                    ) * self.running_mean
            if self.running_var is not None:
                self.running_var = m / (m - 1) * self.momentum * var + (1 -
                    self.momentum) * self.running_var
        torch.cuda.nvtx.range_pop()
        out = SyncBatchnormFunction.apply(input, self.weight, self.bias,
            mean, var, self.eps, process_group, world_size)
    return out.to(cast)
