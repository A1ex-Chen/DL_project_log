def deepview_input_provider(batch_size=32):
    return torch.randn((batch_size, 3, 128, 128)).cuda(),
