def forward(self, *inputs, **kwargs):
    if not self.device_ids:
        return self.module(*inputs, **kwargs)
    inputs = merge_tDBN_batch_training(*inputs)
    inputs = example_convert_to_torch_training(inputs)
    if len(self.device_ids) == 1:
        return self.module(*inputs[0], **kwargs[0])
    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
    outputs = self.parallel_apply(replicas, inputs, ({}, {}))
    return self.gather(outputs, self.output_device)
