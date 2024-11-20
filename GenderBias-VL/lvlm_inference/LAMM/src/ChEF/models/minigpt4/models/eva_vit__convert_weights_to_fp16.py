def _convert_weights_to_fp16(l):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()
