@classmethod
def init_Qformer(cls, num_query_token, vision_width, freeze):
    encoder_config = BertConfig.from_pretrained('bert-base-uncased')
    encoder_config.encoder_width = vision_width
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = 2
    encoder_config.query_length = num_query_token
    Qformer = BertLMHeadModel(config=encoder_config)
    query_tokens = nn.Parameter(torch.zeros(1, num_query_token,
        encoder_config.hidden_size))
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    Qformer.cls = None
    Qformer.bert.embeddings.word_embeddings = None
    Qformer.bert.embeddings.position_embeddings = None
    for layer in Qformer.bert.encoder.layer:
        layer.output = None
        layer.intermediate = None
    if freeze:
        for name, param in Qformer.named_parameters():
            param.requires_grad = False
        Qformer = Qformer.eval()
        Qformer.train = disabled_train
        query_tokens.requires_grad = False
        logging.info('freeze Qformer')
    return Qformer, query_tokens
