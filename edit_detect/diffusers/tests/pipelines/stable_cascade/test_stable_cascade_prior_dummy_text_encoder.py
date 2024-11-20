@property
def dummy_text_encoder(self):
    torch.manual_seed(0)
    config = CLIPTextConfig(bos_token_id=0, eos_token_id=2, hidden_size=
        self.text_embedder_hidden_size, projection_dim=self.
        text_embedder_hidden_size, intermediate_size=37, layer_norm_eps=
        1e-05, num_attention_heads=4, num_hidden_layers=5, pad_token_id=1,
        vocab_size=1000)
    return CLIPTextModelWithProjection(config).eval()
