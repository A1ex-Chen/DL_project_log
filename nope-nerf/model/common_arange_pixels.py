def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1.0, 
    1.0), device=torch.device('cpu')):
    """ Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        device (torch.device): device to use
    """
    h, w = resolution
    pixel_locations = torch.meshgrid(torch.arange(0, h, device=device),
        torch.arange(0, w, device=device))
    pixel_locations = torch.stack([pixel_locations[1], pixel_locations[0]],
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()
    scale = image_range[1] - image_range[0]
    loc = (image_range[1] - image_range[0]) / 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc
    return pixel_locations, pixel_scaled
