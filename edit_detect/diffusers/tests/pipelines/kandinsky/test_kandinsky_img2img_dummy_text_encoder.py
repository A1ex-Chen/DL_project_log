@property
def dummy_text_encoder(self):
    torch.manual_seed(0)
    config = MCLIPConfig(numDims=self.cross_attention_dim,
        transformerDimensions=self.text_embedder_hidden_size, hidden_size=
        self.text_embedder_hidden_size, intermediate_size=37,
        num_attention_heads=4, num_hidden_layers=5, vocab_size=1005)
    text_encoder = MultilingualCLIP(config)
    text_encoder = text_encoder.eval()
    return text_encoder
