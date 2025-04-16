def get_inst_idx_to_tensor_position_map(inst_idx_list):
    """ Indicate the position of an instance in a tensor. """
    return {inst_idx: tensor_position for tensor_position, inst_idx in
        enumerate(inst_idx_list)}
