def model_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'],
        default='best')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-trace_to', type=str)
    parser.add_argument('-seed', type=int)
    parser.add_argument('-load_traced', type=str)
    parser.add_argument('-measure', action='store_true')
    opt = parser.parse_args(args=[])
    opt.seed = 1
    opt.embs_share_weight = False
    opt.proj_share_weight = True
    opt.label_smoothing = True
    opt.d_word_vec = opt.d_model
    opt.max_token_seq_len = 52
    opt.src_vocab_size = 32317
    opt.tgt_vocab_size = 32317
    opt.warm_up = 10
    opt.measure_for = 100
    return opt
