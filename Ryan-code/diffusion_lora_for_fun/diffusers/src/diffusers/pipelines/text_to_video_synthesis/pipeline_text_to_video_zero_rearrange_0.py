def rearrange_0(tensor, f):
    F, C, H, W = tensor.size()
    tensor = torch.permute(torch.reshape(tensor, (F // f, f, C, H, W)), (0,
        2, 1, 3, 4))
    return tensor
