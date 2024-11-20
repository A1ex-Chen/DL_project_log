def get_forward_arguments(self, query_dim, added_kv_proj_dim):
    batch_size = 2
    hidden_states = torch.rand(batch_size, query_dim, 3, 2)
    encoder_hidden_states = torch.rand(batch_size, 4, added_kv_proj_dim)
    attention_mask = None
    return {'hidden_states': hidden_states, 'encoder_hidden_states':
        encoder_hidden_states, 'attention_mask': attention_mask}
