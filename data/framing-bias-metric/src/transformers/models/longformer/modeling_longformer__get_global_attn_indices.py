@staticmethod
def _get_global_attn_indices(is_index_global_attn):
    """ compute global attn indices required throughout forward pass """
    num_global_attn_indices = is_index_global_attn.long().sum(dim=1)
    max_num_global_attn_indices = num_global_attn_indices.max()
    is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)
    is_local_index_global_attn = torch.arange(max_num_global_attn_indices,
        device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)
    is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(
        as_tuple=True)
    is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0
        ).nonzero(as_tuple=True)
    return (max_num_global_attn_indices, is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero)
