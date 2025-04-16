def get_hook(grad_fn):

    def hook(arg1, arg2):
        if not isinstance(arg2[0], torch.Tensor):
            return
        input_dict[grad_fn] = arg2[0].size()
    return hook
