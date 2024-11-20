def smart_inference_mode(torch_1_9=check_version(torch.__version__, '1.9.0')):

    def decorate(fn):
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)
    return decorate
