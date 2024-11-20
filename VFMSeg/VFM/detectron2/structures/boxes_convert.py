@staticmethod
def convert(box: _RawBoxType, from_mode: 'BoxMode', to_mode: 'BoxMode'
    ) ->_RawBoxType:
    """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
    if from_mode == to_mode:
        return box
    original_type = type(box)
    is_numpy = isinstance(box, np.ndarray)
    single_box = isinstance(box, (list, tuple))
    if single_box:
        assert len(box) == 4 or len(box
            ) == 5, 'BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor, where k == 4 or 5'
        arr = torch.tensor(box)[None, :]
    elif is_numpy:
        arr = torch.from_numpy(np.asarray(box)).clone()
    else:
        arr = box.clone()
    assert to_mode not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL
        ] and from_mode not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL
        ], 'Relative mode not yet supported!'
    if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
        assert arr.shape[-1
            ] == 5, 'The last dimension of input shape must be 5 for XYWHA format'
        original_dtype = arr.dtype
        arr = arr.double()
        w = arr[:, 2]
        h = arr[:, 3]
        a = arr[:, 4]
        c = torch.abs(torch.cos(a * math.pi / 180.0))
        s = torch.abs(torch.sin(a * math.pi / 180.0))
        new_w = c * w + s * h
        new_h = c * h + s * w
        arr[:, 0] -= new_w / 2.0
        arr[:, 1] -= new_h / 2.0
        arr[:, 2] = arr[:, 0] + new_w
        arr[:, 3] = arr[:, 1] + new_h
        arr = arr[:, :4].to(dtype=original_dtype)
    elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
        original_dtype = arr.dtype
        arr = arr.double()
        arr[:, 0] += arr[:, 2] / 2.0
        arr[:, 1] += arr[:, 3] / 2.0
        angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
        arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
    elif to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
        arr[:, 2] += arr[:, 0]
        arr[:, 3] += arr[:, 1]
    elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
        arr[:, 2] -= arr[:, 0]
        arr[:, 3] -= arr[:, 1]
    else:
        raise NotImplementedError(
            'Conversion from BoxMode {} to {} is not supported yet'.format(
            from_mode, to_mode))
    if single_box:
        return original_type(arr.flatten().tolist())
    if is_numpy:
        return arr.numpy()
    else:
        return arr
