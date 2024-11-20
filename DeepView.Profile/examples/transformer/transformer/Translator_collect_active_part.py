def collect_active_part(beamed_tensor, curr_active_inst_idx,
    n_prev_active_inst, n_bm):
    """ Collect tensor parts associated to active instances. """
    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = n_curr_active_inst * n_bm, *d_hs
    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)
    return beamed_tensor
