@staticmethod
def boxlist_to_tensor(boxlist, output_box_dim):
    if type(boxlist) == np.ndarray:
        box_tensor = torch.from_numpy(boxlist)
    elif type(boxlist) == list:
        if boxlist == []:
            return torch.zeros((0, output_box_dim), dtype=torch.float32)
        else:
            box_tensor = torch.FloatTensor(boxlist)
    else:
        raise Exception('Unrecognized boxlist type')
    input_box_dim = box_tensor.shape[1]
    if input_box_dim != output_box_dim:
        if input_box_dim == 4 and output_box_dim == 5:
            box_tensor = BoxMode.convert(box_tensor, BoxMode.XYWH_ABS,
                BoxMode.XYWHA_ABS)
        else:
            raise Exception('Unable to convert from {}-dim box to {}-dim box'
                .format(input_box_dim, output_box_dim))
    return box_tensor
