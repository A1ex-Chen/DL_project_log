def forward(self, *inputs, **kwargs):
    if not self.device_ids:
        return self.module(*inputs, **kwargs)
    if self.gpu0_bsz == 0:
        device_ids = self.device_ids[1:]
    else:
        device_ids = self.device_ids
    inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
    if len(self.device_ids) == 1:
        return self.module(*inputs[0], **kwargs[0])
    replicas = self.replicate(self.module, self.device_ids)
    if self.gpu0_bsz == 0:
        replicas = replicas[1:]
    outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
    return self.gather(outputs, self.output_device)
