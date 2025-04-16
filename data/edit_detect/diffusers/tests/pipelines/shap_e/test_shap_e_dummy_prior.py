@property
def dummy_prior(self):
    torch.manual_seed(0)
    model_kwargs = {'num_attention_heads': 2, 'attention_head_dim': 16,
        'embedding_dim': self.time_input_dim, 'num_embeddings': 32,
        'embedding_proj_dim': self.text_embedder_hidden_size,
        'time_embed_dim': self.time_embed_dim, 'num_layers': 1,
        'clip_embed_dim': self.time_input_dim * 2, 'additional_embeddings':
        0, 'time_embed_act_fn': 'gelu', 'norm_in_type': 'layer',
        'encoder_hid_proj_type': None, 'added_emb_type': None}
    model = PriorTransformer(**model_kwargs)
    return model
