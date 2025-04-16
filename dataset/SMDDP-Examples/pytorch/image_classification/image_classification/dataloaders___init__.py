def __init__(self, batch_size, num_classes, num_channels, height, width,
    one_hot, memory_format=torch.contiguous_format):
    input_data = torch.randn(batch_size, num_channels, height, width
        ).contiguous(memory_format=memory_format).cuda().normal_(0, 1.0)
    if one_hot:
        input_target = torch.empty(batch_size, num_classes).cuda()
        input_target[:, 0] = 1.0
    else:
        input_target = torch.randint(0, num_classes, (batch_size,))
    input_target = input_target.cuda()
    self.input_data = input_data
    self.input_target = input_target
