def get_model_device(torch_model):
    if next(torch_model.parameters()).is_cuda:
        return 'cuda'
    else:
        return 'cpu'
