def limit_reg_target(boxes):
    """
        limit the reg_target to [-1/4*np.pi, 1/4*np.pi],
    """
    boxes[..., -1] += torch.where(boxes[..., -1:] < -3.0 / 4.0 * np.pi,
        torch.tensor(np.pi).type_as(boxes), torch.tensor(0.0).type_as(boxes))[
        ..., 0]
    boxes[..., -1] -= torch.where(boxes[..., -1:] > 3.0 / 4.0 * np.pi,
        torch.tensor(np.pi).type_as(boxes), torch.tensor(0.0).type_as(boxes))[
        ..., 0]
    return boxes
