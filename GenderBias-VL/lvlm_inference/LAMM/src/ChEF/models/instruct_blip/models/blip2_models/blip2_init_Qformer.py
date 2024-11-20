@classmethod
def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
    encoder_config = BertConfig.from_pretrained('bert-base-uncased')
    encoder_config.encoder_width = vision_width
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    encoder_config.query_length = num_query_token
    Qformer = BertLMHeadModel.from_pretrained('bert-base-uncased', config=
        encoder_config)
    query_tokens = nn.Parameter(torch.zeros(1, num_query_token,
        encoder_config.hidden_size))
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return Qformer, query_tokens
