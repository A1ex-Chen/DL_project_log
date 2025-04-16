def deepview_model_provider():
    opt = model_config()
    return TransformerWithLoss(Transformer(opt.src_vocab_size, opt.
        tgt_vocab_size, opt.max_token_seq_len, tgt_emb_prj_weight_sharing=
        opt.proj_share_weight, emb_src_tgt_weight_sharing=opt.
        embs_share_weight, d_k=opt.d_k, d_v=opt.d_v, d_model=opt.d_model,
        d_word_vec=opt.d_word_vec, d_inner=opt.d_inner_hid, n_layers=opt.
        n_layers, n_head=opt.n_head, dropout=opt.dropout)).cuda()
