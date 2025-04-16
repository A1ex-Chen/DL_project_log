def foo(net):
    childrens = list(net.children())
    if not childrens:
        if isinstance(net, nn.Conv2d):
            net.register_forward_hook(conv2d_hook)
        elif isinstance(net, nn.Conv1d):
            net.register_forward_hook(conv1d_hook)
        elif isinstance(net, nn.Linear):
            net.register_forward_hook(linear_hook)
        elif isinstance(net, nn.BatchNorm2d) or isinstance(net, nn.BatchNorm1d
            ):
            net.register_forward_hook(bn_hook)
        elif isinstance(net, nn.ReLU):
            net.register_forward_hook(relu_hook)
        elif isinstance(net, nn.AvgPool2d) or isinstance(net, nn.MaxPool2d):
            net.register_forward_hook(pooling2d_hook)
        elif isinstance(net, nn.AvgPool1d) or isinstance(net, nn.MaxPool1d):
            net.register_forward_hook(pooling1d_hook)
        else:
            print('Warning: flop of module {} is not counted!'.format(net))
        return
    for c in childrens:
        foo(c)
