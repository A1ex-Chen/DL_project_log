def poly_area_tensor(x, y):
    return 0.5 * torch.abs(torch.dot(x, torch.roll(y, 1)) - torch.dot(y,
        torch.roll(x, 1)))
