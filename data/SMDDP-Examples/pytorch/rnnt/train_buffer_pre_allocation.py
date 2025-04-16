def buffer_pre_allocation():
    max_seq_len = math.ceil(train_preproc.audio_duration_to_seq_len(cfg[
        'input_train']['audio_dataset']['max_duration'], after_subsampling=
        False, after_stack_time=False) * cfg['input_train']['audio_dataset'
        ]['speed_perturbation']['max_rate'])
    max_txt_len = train_loader.data_iterator().max_txt_len
    print_once(
        f'Pre-allocate buffer with max_seq_len of %d and max_txt_len of %d' %
        (max_seq_len, max_txt_len))
    audio = torch.ones(batch_size, cfg['input_val']['filterbank_features'][
        'n_filt'], max_seq_len, dtype=torch.float32, device='cuda')
    audio_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda'
        ) * max_seq_len
    txt = torch.ones(batch_size, max_txt_len, dtype=torch.int64, device='cuda')
    txt_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda'
        ) * max_txt_len
    feats, feat_lens = train_feat_proc([audio, audio_lens])
    if args.dist_lamb:
        feats = feats.half()
    meta_data = []
    B_split = batch_size // args.batch_split_factor
    for i in range(args.batch_split_factor):
        meta_data.append(train_preproc.get_packing_meta_data(feats.size(0),
            feat_lens[i * B_split:(i + 1) * B_split], txt_lens[i * B_split:
            (i + 1) * B_split]))
    train_step(model, loss_fn, args, batch_size, feats, feat_lens, txt,
        txt_lens, optimizer, grad_scaler, meta_data, None, rnnt_graph,
        copy_stream, pred_stream)
