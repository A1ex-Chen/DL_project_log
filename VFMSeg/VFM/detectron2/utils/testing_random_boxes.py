def random_boxes(num_boxes, max_coord=100, device='cpu'):
    """
    Create a random Nx4 boxes tensor, with coordinates < max_coord.
    """
    boxes = torch.rand(num_boxes, 4, device=device) * (max_coord * 0.5)
    boxes.clamp_(min=1.0)
    boxes[:, 2:] += boxes[:, :2]
    return boxes
