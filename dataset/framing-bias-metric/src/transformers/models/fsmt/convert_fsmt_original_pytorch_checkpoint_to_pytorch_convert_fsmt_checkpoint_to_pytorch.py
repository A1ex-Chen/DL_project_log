def convert_fsmt_checkpoint_to_pytorch(fsmt_checkpoint_path,
    pytorch_dump_folder_path):
    assert os.path.exists(fsmt_checkpoint_path)
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    print(f'Writing results to {pytorch_dump_folder_path}')
    checkpoint_file = basename(fsmt_checkpoint_path)
    fsmt_folder_path = dirname(fsmt_checkpoint_path)
    cls = (fairseq.model_parallel.models.transformer.
        ModelParallelTransformerModel)
    models = cls.hub_models()
    kwargs = {'bpe': 'fastbpe', 'tokenizer': 'moses'}
    data_name_or_path = '.'
    print(f'using checkpoint {checkpoint_file}')
    chkpt = hub_utils.from_pretrained(fsmt_folder_path, checkpoint_file,
        data_name_or_path, archive_map=models, **kwargs)
    args = vars(chkpt['args']['model'])
    src_lang = args['source_lang']
    tgt_lang = args['target_lang']
    data_root = dirname(pytorch_dump_folder_path)
    model_dir = basename(pytorch_dump_folder_path)
    src_dict_file = os.path.join(fsmt_folder_path, f'dict.{src_lang}.txt')
    tgt_dict_file = os.path.join(fsmt_folder_path, f'dict.{tgt_lang}.txt')
    src_dict = Dictionary.load(src_dict_file)
    src_vocab = rewrite_dict_keys(src_dict.indices)
    src_vocab_size = len(src_vocab)
    src_vocab_file = os.path.join(pytorch_dump_folder_path, 'vocab-src.json')
    print(
        f'Generating {src_vocab_file} of {src_vocab_size} of {src_lang} records'
        )
    with open(src_vocab_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(src_vocab, ensure_ascii=False, indent=json_indent))
    do_lower_case = True
    for k in src_vocab.keys():
        if not k.islower():
            do_lower_case = False
            break
    tgt_dict = Dictionary.load(tgt_dict_file)
    tgt_vocab = rewrite_dict_keys(tgt_dict.indices)
    tgt_vocab_size = len(tgt_vocab)
    tgt_vocab_file = os.path.join(pytorch_dump_folder_path, 'vocab-tgt.json')
    print(
        f'Generating {tgt_vocab_file} of {tgt_vocab_size} of {tgt_lang} records'
        )
    with open(tgt_vocab_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tgt_vocab, ensure_ascii=False, indent=json_indent))
    merges_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES[
        'merges_file'])
    for fn in ['bpecodes', 'code']:
        fsmt_merges_file = os.path.join(fsmt_folder_path, fn)
        if os.path.exists(fsmt_merges_file):
            break
    with open(fsmt_merges_file, encoding='utf-8') as fin:
        merges = fin.read()
    merges = re.sub(' \\d+$', '', merges, 0, re.M)
    print(f'Generating {merges_file}')
    with open(merges_file, 'w', encoding='utf-8') as fout:
        fout.write(merges)
    fsmt_model_config_file = os.path.join(pytorch_dump_folder_path,
        'config.json')
    assert args['bpe'
        ] == 'fastbpe', f"need to extend tokenizer to support bpe={args['bpe']}"
    assert args['tokenizer'
        ] == 'moses', f"need to extend tokenizer to support bpe={args['tokenizer']}"
    model_conf = {'architectures': ['FSMTForConditionalGeneration'],
        'model_type': 'fsmt', 'activation_dropout': args[
        'activation_dropout'], 'activation_function': 'relu',
        'attention_dropout': args['attention_dropout'], 'd_model': args[
        'decoder_embed_dim'], 'dropout': args['dropout'], 'init_std': 0.02,
        'max_position_embeddings': args['max_source_positions'],
        'num_hidden_layers': args['encoder_layers'], 'src_vocab_size':
        src_vocab_size, 'tgt_vocab_size': tgt_vocab_size, 'langs': [
        src_lang, tgt_lang], 'encoder_attention_heads': args[
        'encoder_attention_heads'], 'encoder_ffn_dim': args[
        'encoder_ffn_embed_dim'], 'encoder_layerdrop': args[
        'encoder_layerdrop'], 'encoder_layers': args['encoder_layers'],
        'decoder_attention_heads': args['decoder_attention_heads'],
        'decoder_ffn_dim': args['decoder_ffn_embed_dim'],
        'decoder_layerdrop': args['decoder_layerdrop'], 'decoder_layers':
        args['decoder_layers'], 'bos_token_id': 0, 'pad_token_id': 1,
        'eos_token_id': 2, 'is_encoder_decoder': True, 'scale_embedding': 
        not args['no_scale_embedding'], 'tie_word_embeddings': args[
        'share_all_embeddings']}
    model_conf['num_beams'] = 5
    model_conf['early_stopping'] = False
    if (model_dir in best_score_hparams and 'length_penalty' in
        best_score_hparams[model_dir]):
        model_conf['length_penalty'] = best_score_hparams[model_dir][
            'length_penalty']
    else:
        model_conf['length_penalty'] = 1.0
    print(f'Generating {fsmt_model_config_file}')
    with open(fsmt_model_config_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(model_conf, ensure_ascii=False, indent=json_indent))
    fsmt_tokenizer_config_file = os.path.join(pytorch_dump_folder_path,
        TOKENIZER_CONFIG_FILE)
    tokenizer_conf = {'langs': [src_lang, tgt_lang], 'model_max_length': 
        1024, 'do_lower_case': do_lower_case}
    print(f'Generating {fsmt_tokenizer_config_file}')
    with open(fsmt_tokenizer_config_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_conf, ensure_ascii=False, indent=
            json_indent))
    model = chkpt['models'][0]
    model_state_dict = model.state_dict()
    model_state_dict = OrderedDict(('model.' + k, v) for k, v in
        model_state_dict.items())
    ignore_keys = ['model.model', 'model.encoder.version',
        'model.decoder.version', 'model.encoder_embed_tokens.weight',
        'model.decoder_embed_tokens.weight',
        'model.encoder.embed_positions._float_tensor',
        'model.decoder.embed_positions._float_tensor']
    for k in ignore_keys:
        model_state_dict.pop(k, None)
    config = FSMTConfig.from_pretrained(pytorch_dump_folder_path)
    model_new = FSMTForConditionalGeneration(config)
    model_new.load_state_dict(model_state_dict, strict=False)
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path,
        WEIGHTS_NAME)
    print(f'Generating {pytorch_weights_dump_path}')
    torch.save(model_state_dict, pytorch_weights_dump_path)
    print('Conversion is done!')
    print('\nLast step is to upload the files to s3')
    print(f'cd {data_root}')
    print(f'transformers-cli upload {model_dir}')
