def glue_convert_examples_to_features(examples, tokenizer, max_length=512,
    task=None, label_list=None, output_mode=None, pad_on_left=False,
    pad_token=0, pad_token_segment_id=0, mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True
    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info('Using label list %s for task %s' % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info('Using output mode %s for task %s' % (output_mode,
                task))
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info('Writing example %d' % ex_index)
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
        inputs = tokenizer.encode_plus(example.text_a, example.text_b,
            add_special_tokens=True, max_length=max_length)
        input_ids, token_type_ids = inputs['input_ids'], inputs[
            'token_type_ids']
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = [pad_token] * padding_length + input_ids
            attention_mask = [0 if mask_padding_with_zero else 1
                ] * padding_length + attention_mask
            token_type_ids = [pad_token_segment_id
                ] * padding_length + token_type_ids
        else:
            input_ids = input_ids + [pad_token] * padding_length
            attention_mask = attention_mask + [0 if mask_padding_with_zero else
                1] * padding_length
            token_type_ids = token_type_ids + [pad_token_segment_id
                ] * padding_length
        assert len(input_ids
            ) == max_length, 'Error with input length {} vs {}'.format(len(
            input_ids), max_length)
        assert len(attention_mask
            ) == max_length, 'Error with input length {} vs {}'.format(len(
            attention_mask), max_length)
        assert len(token_type_ids
            ) == max_length, 'Error with input length {} vs {}'.format(len(
            token_type_ids), max_length)
        if output_mode == 'classification':
            label = label_map[example.label]
        elif output_mode == 'regression':
            label = float(example.label)
        else:
            raise KeyError(output_mode)
        if ex_index < 5:
            logger.info('*** Example ***')
            logger.info('guid: %s' % example.guid)
            logger.info('input_ids: %s' % ' '.join([str(x) for x in input_ids])
                )
            logger.info('attention_mask: %s' % ' '.join([str(x) for x in
                attention_mask]))
            logger.info('token_type_ids: %s' % ' '.join([str(x) for x in
                token_type_ids]))
            logger.info('label: %s (id = %d)' % (example.label, label))
        features.append(InputFeatures(input_ids=input_ids, attention_mask=
            attention_mask, token_type_ids=token_type_ids, label=label))
    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield {'input_ids': ex.input_ids, 'attention_mask': ex.
                    attention_mask, 'token_type_ids': ex.token_type_ids
                    }, ex.label
        return tf.data.Dataset.from_generator(gen, ({'input_ids': tf.int32,
            'attention_mask': tf.int32, 'token_type_ids': tf.int32}, tf.
            int64), ({'input_ids': tf.TensorShape([None]), 'attention_mask':
            tf.TensorShape([None]), 'token_type_ids': tf.TensorShape([None]
            )}, tf.TensorShape([])))
    return features
