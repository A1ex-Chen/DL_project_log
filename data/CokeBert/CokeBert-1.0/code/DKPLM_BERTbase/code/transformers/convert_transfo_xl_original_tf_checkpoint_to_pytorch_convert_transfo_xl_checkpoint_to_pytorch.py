def convert_transfo_xl_checkpoint_to_pytorch(tf_checkpoint_path,
    transfo_xl_config_file, pytorch_dump_folder_path, transfo_xl_dataset_file):
    if transfo_xl_dataset_file:
        with open(transfo_xl_dataset_file, 'rb') as fp:
            corpus = pickle.load(fp, encoding='latin1')
        pytorch_vocab_dump_path = (pytorch_dump_folder_path + '/' +
            VOCAB_FILES_NAMES['pretrained_vocab_file'])
        print('Save vocabulary to {}'.format(pytorch_vocab_dump_path))
        corpus_vocab_dict = corpus.vocab.__dict__
        torch.save(corpus_vocab_dict, pytorch_vocab_dump_path)
        corpus_dict_no_vocab = corpus.__dict__
        corpus_dict_no_vocab.pop('vocab', None)
        pytorch_dataset_dump_path = (pytorch_dump_folder_path + '/' +
            CORPUS_NAME)
        print('Save dataset to {}'.format(pytorch_dataset_dump_path))
        torch.save(corpus_dict_no_vocab, pytorch_dataset_dump_path)
    if tf_checkpoint_path:
        config_path = os.path.abspath(transfo_xl_config_file)
        tf_path = os.path.abspath(tf_checkpoint_path)
        print('Converting Transformer XL checkpoint from {} with config at {}'
            .format(tf_path, config_path))
        if transfo_xl_config_file == '':
            config = TransfoXLConfig()
        else:
            config = TransfoXLConfig.from_json_file(transfo_xl_config_file)
        print('Building PyTorch model from configuration: {}'.format(str(
            config)))
        model = TransfoXLLMHeadModel(config)
        model = load_tf_weights_in_transfo_xl(model, config, tf_path)
        pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path,
            WEIGHTS_NAME)
        pytorch_config_dump_path = os.path.join(pytorch_dump_folder_path,
            CONFIG_NAME)
        print('Save PyTorch model to {}'.format(os.path.abspath(
            pytorch_weights_dump_path)))
        torch.save(model.state_dict(), pytorch_weights_dump_path)
        print('Save configuration file to {}'.format(os.path.abspath(
            pytorch_config_dump_path)))
        with open(pytorch_config_dump_path, 'w', encoding='utf-8') as f:
            f.write(config.to_json_string())
