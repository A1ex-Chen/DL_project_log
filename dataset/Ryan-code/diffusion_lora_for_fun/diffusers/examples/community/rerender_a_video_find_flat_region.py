@torch.no_grad()
def find_flat_region(mask):
    device = mask.device
    kernel_x = torch.Tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).unsqueeze(0
        ).unsqueeze(0).to(device)
    kernel_y = torch.Tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).unsqueeze(0
        ).unsqueeze(0).to(device)
    mask_ = F.pad(mask.unsqueeze(0), (1, 1, 1, 1), mode='replicate')
    grad_x = torch.nn.functional.conv2d(mask_, kernel_x)
    grad_y = torch.nn.functional.conv2d(mask_, kernel_y)
    return (abs(grad_x) + abs(grad_y) == 0).float()[0]
