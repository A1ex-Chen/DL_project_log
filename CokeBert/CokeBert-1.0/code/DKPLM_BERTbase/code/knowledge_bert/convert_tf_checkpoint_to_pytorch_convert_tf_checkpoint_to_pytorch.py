def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file,
    pytorch_dump_path):
    config_path = os.path.abspath(bert_config_file)
    tf_path = os.path.abspath(tf_checkpoint_path)
    print('Converting TensorFlow checkpoint from {} with config at {}'.
        format(tf_path, config_path))
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    config = BertConfig.from_json_file(bert_config_file)
    print('Building PyTorch model from configuration: {}'.format(str(config)))
    model = BertForPreTraining(config)
    for name, array in zip(names, arrays):
        name = name.split('/')
        if any(n in ['adam_v', 'adam_m'] for n in name):
            print('Skipping {}'.format('/'.join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                l = re.split('_(\\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        print('Initialize PyTorch weight {}'.format(name))
        pointer.data = torch.from_numpy(array)
    print('Save PyTorch model to {}'.format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)
