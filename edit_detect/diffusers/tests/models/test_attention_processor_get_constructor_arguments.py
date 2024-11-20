def get_constructor_arguments(self, only_cross_attention: bool=False):
    query_dim = 10
    if only_cross_attention:
        cross_attention_dim = 12
    else:
        cross_attention_dim = query_dim
    return {'query_dim': query_dim, 'cross_attention_dim':
        cross_attention_dim, 'heads': 2, 'dim_head': 4, 'added_kv_proj_dim':
        6, 'norm_num_groups': 1, 'only_cross_attention':
        only_cross_attention, 'processor': AttnAddedKVProcessor()}
