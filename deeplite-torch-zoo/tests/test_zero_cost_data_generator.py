def data_generator(model, shuffle_data=True, input_gradient=False):
    test_loader = cifar100_dataloaders['test'] if shuffle_data else repeat(next
        (iter(cifar100_dataloaders['test'])))
    for inputs, targets in test_loader:
        inputs.requires_grad_(input_gradient)
        outputs = model(inputs)
        yield inputs, outputs, targets, {'loss_kwarg': None}
