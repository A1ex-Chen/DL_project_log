def load_tf2_weights_in_bert(model, tf_checkpoint_path, config):
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info('Converting TensorFlow checkpoint from {}'.format(tf_path))
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    layer_depth = []
    for full_name, shape in init_vars:
        name = full_name.split('/')
        if full_name == '_CHECKPOINTABLE_OBJECT_GRAPH' or name[0] in [
            'global_step', 'save_counter']:
            logger.info(f'Skipping non-model layer {full_name}')
            continue
        if 'optimizer' in full_name:
            logger.info(f'Skipping optimization layer {full_name}')
            continue
        if name[0] == 'model':
            name = name[1:]
        depth = 0
        for _name in name:
            if _name.startswith('layer_with_weights'):
                depth += 1
            else:
                break
        layer_depth.append(depth)
        array = tf.train.load_variable(tf_path, full_name)
        names.append('/'.join(name))
        arrays.append(array)
    logger.info(f'Read a total of {len(arrays):,} layers')
    if len(set(layer_depth)) != 1:
        raise ValueError(
            f'Found layer names with different depths (layer depth {list(set(layer_depth))})'
            )
    layer_depth = list(set(layer_depth))[0]
    if layer_depth != 1:
        raise ValueError(
            'The model contains more than just the embedding/encoder layers. This script does not handle MLM/NSP heads.'
            )
    logger.info('Converting weights...')
    for full_name, array in zip(names, arrays):
        name = full_name.split('/')
        pointer = model
        trace = []
        for i, m_name in enumerate(name):
            if m_name == '.ATTRIBUTES':
                break
            if m_name.startswith('layer_with_weights'):
                layer_num = int(m_name.split('-')[-1])
                if layer_num <= 2:
                    continue
                elif layer_num == 3:
                    trace.extend(['embeddings', 'LayerNorm'])
                    pointer = getattr(pointer, 'embeddings')
                    pointer = getattr(pointer, 'LayerNorm')
                elif layer_num > 3 and layer_num < config.num_hidden_layers + 4:
                    trace.extend(['encoder', 'layer', str(layer_num - 4)])
                    pointer = getattr(pointer, 'encoder')
                    pointer = getattr(pointer, 'layer')
                    pointer = pointer[layer_num - 4]
                elif layer_num == config.num_hidden_layers + 4:
                    trace.extend(['pooler', 'dense'])
                    pointer = getattr(pointer, 'pooler')
                    pointer = getattr(pointer, 'dense')
            elif m_name == 'embeddings':
                trace.append('embeddings')
                pointer = getattr(pointer, 'embeddings')
                if layer_num == 0:
                    trace.append('word_embeddings')
                    pointer = getattr(pointer, 'word_embeddings')
                elif layer_num == 1:
                    trace.append('position_embeddings')
                    pointer = getattr(pointer, 'position_embeddings')
                elif layer_num == 2:
                    trace.append('token_type_embeddings')
                    pointer = getattr(pointer, 'token_type_embeddings')
                else:
                    raise ValueError(
                        'Unknown embedding layer with name {full_name}')
                trace.append('weight')
                pointer = getattr(pointer, 'weight')
            elif m_name == '_attention_layer':
                trace.extend(['attention', 'self'])
                pointer = getattr(pointer, 'attention')
                pointer = getattr(pointer, 'self')
            elif m_name == '_attention_layer_norm':
                trace.extend(['attention', 'output', 'LayerNorm'])
                pointer = getattr(pointer, 'attention')
                pointer = getattr(pointer, 'output')
                pointer = getattr(pointer, 'LayerNorm')
            elif m_name == '_attention_output_dense':
                trace.extend(['attention', 'output', 'dense'])
                pointer = getattr(pointer, 'attention')
                pointer = getattr(pointer, 'output')
                pointer = getattr(pointer, 'dense')
            elif m_name == '_output_dense':
                trace.extend(['output', 'dense'])
                pointer = getattr(pointer, 'output')
                pointer = getattr(pointer, 'dense')
            elif m_name == '_output_layer_norm':
                trace.extend(['output', 'LayerNorm'])
                pointer = getattr(pointer, 'output')
                pointer = getattr(pointer, 'LayerNorm')
            elif m_name == '_key_dense':
                trace.append('key')
                pointer = getattr(pointer, 'key')
            elif m_name == '_query_dense':
                trace.append('query')
                pointer = getattr(pointer, 'query')
            elif m_name == '_value_dense':
                trace.append('value')
                pointer = getattr(pointer, 'value')
            elif m_name == '_intermediate_dense':
                trace.extend(['intermediate', 'dense'])
                pointer = getattr(pointer, 'intermediate')
                pointer = getattr(pointer, 'dense')
            elif m_name == '_output_layer_norm':
                trace.append('output')
                pointer = getattr(pointer, 'output')
            elif m_name in ['bias', 'beta']:
                trace.append('bias')
                pointer = getattr(pointer, 'bias')
            elif m_name in ['kernel', 'gamma']:
                trace.append('weight')
                pointer = getattr(pointer, 'weight')
            else:
                logger.warning(f'Ignored {m_name}')
        trace = '.'.join(trace)
        if re.match(
            '(\\S+)\\.attention\\.self\\.(key|value|query)\\.(bias|weight)',
            trace) or re.match('(\\S+)\\.attention\\.output\\.dense\\.weight',
            trace):
            array = array.reshape(pointer.data.shape)
        if 'kernel' in full_name:
            array = array.transpose()
        if pointer.shape == array.shape:
            pointer.data = torch.from_numpy(array)
        else:
            raise ValueError(
                f'Shape mismatch in layer {full_name}: Model expects shape {pointer.shape} but layer contains shape: {array.shape}'
                )
        logger.info(
            f'Successfully set variable {full_name} to PyTorch layer {trace}')
    return model
