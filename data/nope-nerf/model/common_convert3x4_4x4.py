def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0, 0, 0, 1]], dtype=
                input.dtype, device=input.device)], dim=0)
    elif len(input.shape) == 3:
        output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)
        output[:, 3, 3] = 1.0
    else:
        output = np.concatenate([input, np.array([[0, 0, 0, 1]], dtype=
            input.dtype)], axis=0)
        output[3, 3] = 1.0
    return output
