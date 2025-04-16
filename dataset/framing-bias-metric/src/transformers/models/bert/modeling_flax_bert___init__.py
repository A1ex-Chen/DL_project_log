def __init__(self, config: BertConfig, state: dict, seed: int=0, **kwargs):
    model = FlaxBertModule(vocab_size=config.vocab_size, hidden_size=config
        .hidden_size, type_vocab_size=config.type_vocab_size, max_length=
        config.max_position_embeddings, num_encoder_layers=config.
        num_hidden_layers, num_heads=config.num_attention_heads, head_size=
        config.hidden_size, intermediate_size=config.intermediate_size)
    super().__init__(config, model, state, seed)
