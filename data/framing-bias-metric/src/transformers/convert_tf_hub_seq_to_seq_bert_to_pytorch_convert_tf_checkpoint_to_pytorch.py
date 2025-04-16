def convert_tf_checkpoint_to_pytorch(tf_hub_path, pytorch_dump_path,
    is_encoder_named_decoder, vocab_size, is_encoder):
    bert_config = BertConfig.from_pretrained('bert-large-cased', vocab_size
        =vocab_size, max_position_embeddings=512, is_decoder=True,
        add_cross_attention=True)
    bert_config_dict = bert_config.to_dict()
    del bert_config_dict['type_vocab_size']
    config = BertGenerationConfig(**bert_config_dict)
    if is_encoder:
        model = BertGenerationEncoder(config)
    else:
        model = BertGenerationDecoder(config)
    print('Building PyTorch model from configuration: {}'.format(str(config)))
    load_tf_weights_in_bert_generation(model, tf_hub_path, model_class=
        'bert', is_encoder_named_decoder=is_encoder_named_decoder,
        is_encoder=is_encoder)
    print('Save PyTorch model and config to {}'.format(pytorch_dump_path))
    model.save_pretrained(pytorch_dump_path)
