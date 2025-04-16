def get_box_scales(boxes: Boxes):
    return torch.sqrt((boxes.tensor[:, 2] - boxes.tensor[:, 0]) * (boxes.
        tensor[:, 3] - boxes.tensor[:, 1]))
