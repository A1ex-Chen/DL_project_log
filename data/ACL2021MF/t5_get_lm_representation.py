def get_lm_representation(config, tokenizer, copy_vocab=None):
    attachable_index = set()
    for index in range(tokenizer.sp_model.get_piece_size()):
        x = tokenizer.sp_model.IdToPiece(index)
        if not x[0] == chr(9601) and not all([(c in string.punctuation) for
            c in x]):
            attachable_index.add(index)
    fg_str_dict = None
    if copy_vocab is not None:
        fg_str_dict = {}
        for fg_index in copy_vocab.token_fg_w:
            fg_ch_list = copy_vocab.token_fg_w[fg_index]
            s1 = '&'.join([str(f) for f in fg_ch_list])
            fg_str_dict[fg_index] = s1
    if config.do_pretrain_lm_init:
        model = T5WithMF.from_pretrained(config.lm_type, return_dict=True,
            cache_dir='.', local_config=config, copy_vocab=copy_vocab,
            attachable_index=attachable_index, fg_str_dict=fg_str_dict)
    else:
        lm_config = T5Config.from_pretrained(config.lm_type, return_dict=
            True, cache_dir='.')
        lm_config.num_layers = 3
        model = T5WithMF(lm_config, local_config=config, copy_vocab=
            copy_vocab, attachable_index=attachable_index, fg_str_dict=
            fg_str_dict)
    if config.freeze_param and config.do_pretrain_lm_init:
        for p in model.shared.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False
        for dec_block in model.decoder.block:
            for p in dec_block.layer[0].parameters():
                p.requires_grad = False
            for p in dec_block.layer[-1].parameters():
                p.requires_grad = False
    enc_block = model.encoder.block[0]
    if config.use_orginal_enc_pos_embs:
        if config.freeze_enc_pos_param:
            for p in enc_block.layer[0
                ].SelfAttention.relative_attention_bias.parameters():
                p.requires_grad = False
    else:
        assert config.relative_pos_num > 20, 'new relative pos embeds should be positive'
        new_relative_attention_bias = np.empty((config.relative_pos_num,
            model.config.num_heads), dtype=np.float32)
        old_relative_attention_bias_np = enc_block.layer[0
            ].SelfAttention.relative_attention_bias.weight.detach().cpu(
            ).numpy()
        for i in range(32):
            new_relative_attention_bias[i] = old_relative_attention_bias_np[i]
        enc_block.layer[0
            ].SelfAttention.relative_attention_bias = nn.Embedding.from_pretrained(
            torch.tensor(new_relative_attention_bias), freeze=False)
    return {'t5': model, 'attachable_index': attachable_index}
