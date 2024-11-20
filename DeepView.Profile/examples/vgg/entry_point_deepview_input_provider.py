def deepview_input_provider(batch_size=16):
    return torch.randn((batch_size, 3, 224, 224)).cuda(), torch.randint(low
        =0, high=1000, size=(batch_size,)).cuda()
