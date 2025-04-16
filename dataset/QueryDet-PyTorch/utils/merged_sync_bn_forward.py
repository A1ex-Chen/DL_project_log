def forward(self, inputs):
    with autocast(False):
        if comm.get_world_size() == 1 or not self.training:
            return self._eval_forward(inputs)
        B, C = inputs[0].shape[0], inputs[0].shape[1]
        mean = sum([torch.mean(input, dim=[0, 2, 3]) for input in inputs]
            ) / len(inputs)
        meansqr = sum([torch.mean(input * input, dim=[0, 2, 3]) for input in
            inputs]) / len(inputs)
        if self._stats_mode == '':
            assert B > 0, 'SyncBatchNorm(stats_mode="") does not support zero batch size.'
            vec = torch.cat([mean, meansqr], dim=0)
            vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())
            mean, meansqr = torch.split(vec, C)
            momentum = self.momentum
        else:
            if B == 0:
                vec = torch.zeros([2 * C + 1], device=mean.device, dtype=
                    mean.dtype)
                vec = vec + _input.sum()
            else:
                vec = torch.cat([mean, meansqr, torch.ones([1], device=mean
                    .device, dtype=mean.dtype)], dim=0)
            vec = AllReduce.apply(vec * B)
            total_batch = vec[-1].detach()
            momentum = total_batch.clamp(max=1) * self.momentum
            total_batch = torch.max(total_batch, torch.ones_like(total_batch))
            mean, meansqr, _ = torch.split(vec / total_batch, C)
        var = meansqr - mean * mean
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        self.running_mean += momentum * (mean.detach() - self.running_mean)
        self.running_var += momentum * (var.detach() - self.running_var)
        self._batch_mean = mean
        self._batch_meansqr = meansqr
        outputs = [(input * scale + bias) for input in inputs]
        return outputs
