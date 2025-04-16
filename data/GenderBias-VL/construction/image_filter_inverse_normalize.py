def inverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
    0.225]):
    tensor = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor
        ) else tensor
    for i in range(tensor.shape[0]):
        tensor[i] = tensor[i] * std[i] + mean[i]
    return tensor
