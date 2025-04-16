@staticmethod
def reload_device(device, model, task):
    if task == 'train':
        device = next(model.parameters()).device
    else:
        if device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        elif device:
            os.environ['CUDA_VISIBLE_DEVICES'] = device
            assert torch.cuda.is_available()
        cuda = device != 'cpu' and torch.cuda.is_available()
        device = torch.device('cuda:0' if cuda else 'cpu')
    return device
