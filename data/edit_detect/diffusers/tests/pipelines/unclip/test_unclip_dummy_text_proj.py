@property
def dummy_text_proj(self):
    torch.manual_seed(0)
    model_kwargs = {'clip_embeddings_dim': self.text_embedder_hidden_size,
        'time_embed_dim': self.time_embed_dim, 'cross_attention_dim': self.
        cross_attention_dim}
    model = UnCLIPTextProjModel(**model_kwargs)
    return model
