@property
def dummy_prior(self):
    torch.manual_seed(0)
    model_kwargs = {'num_attention_heads': 2, 'attention_head_dim': 12,
        'embedding_dim': self.text_embedder_hidden_size, 'num_layers': 1}
    model = PriorTransformer(**model_kwargs)
    return model
