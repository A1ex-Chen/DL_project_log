def get_tf_weights_as_numpy(path='./ckpt/aeslc/model.ckpt-32000') ->Dict:
    init_vars = tf.train.list_variables(path)
    tf_weights = {}
    ignore_name = ['Adafactor', 'global_step']
    for name, shape in tqdm(init_vars, desc='converting tf checkpoint to dict'
        ):
        skip_key = any([(pat in name) for pat in ignore_name])
        if skip_key:
            continue
        array = tf.train.load_variable(path, name)
        tf_weights[name] = array
    return tf_weights
