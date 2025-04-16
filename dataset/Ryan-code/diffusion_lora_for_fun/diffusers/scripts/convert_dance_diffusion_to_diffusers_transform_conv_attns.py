def transform_conv_attns(new_state_dict, new_k, v):
    if len(new_k) == 1:
        if len(v.shape) == 3:
            new_state_dict[new_k[0]] = v[:, :, 0]
        else:
            new_state_dict[new_k[0]] = v
    else:
        trippled_shape = v.shape[0]
        single_shape = trippled_shape // 3
        for i in range(3):
            if len(v.shape) == 3:
                new_state_dict[new_k[i]] = v[i * single_shape:(i + 1) *
                    single_shape, :, 0]
            else:
                new_state_dict[new_k[i]] = v[i * single_shape:(i + 1) *
                    single_shape]
    return new_state_dict
