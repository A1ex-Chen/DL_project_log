def __init__(self, opt):
    self.opt = opt
    self.device = torch.device('cuda' if opt.cuda else 'cpu')
    checkpoint = torch.load(opt.model)
    model_opt = checkpoint['settings']
    self.model_opt = model_opt
    model = Transformer(model_opt.src_vocab_size, model_opt.tgt_vocab_size,
        model_opt.max_token_seq_len, tgt_emb_prj_weight_sharing=model_opt.
        proj_share_weight, emb_src_tgt_weight_sharing=model_opt.
        embs_share_weight, d_k=model_opt.d_k, d_v=model_opt.d_v, d_model=
        model_opt.d_model, d_word_vec=model_opt.d_word_vec, d_inner=
        model_opt.d_inner_hid, n_layers=model_opt.n_layers, n_head=
        model_opt.n_head, dropout=model_opt.dropout)
    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    model.word_prob_prj = nn.LogSoftmax(dim=1)
    model = model.to(self.device)
    self.model = model
    self.model.eval()
