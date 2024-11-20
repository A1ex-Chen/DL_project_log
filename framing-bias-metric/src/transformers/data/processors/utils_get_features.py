def get_features(self, tokenizer, max_length=None, pad_on_left=False,
    pad_token=0, mask_padding_with_zero=True, return_tensors=None):
    """
        Convert examples in a list of ``InputFeatures``

        Args:
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token: Padding token
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)

        Returns:
            If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset`` containing the
            task-specific features. If the input is a list of ``InputExamples``, will return a list of task-specific
            ``InputFeatures`` which can be fed to the model.

        """
    if max_length is None:
        max_length = tokenizer.max_len
    label_map = {label: i for i, label in enumerate(self.labels)}
    all_input_ids = []
    for ex_index, example in enumerate(self.examples):
        if ex_index % 10000 == 0:
            logger.info('Tokenizing example %d', ex_index)
        input_ids = tokenizer.encode(example.text_a, add_special_tokens=
            True, max_length=min(max_length, tokenizer.max_len))
        all_input_ids.append(input_ids)
    batch_length = max(len(input_ids) for input_ids in all_input_ids)
    features = []
    for ex_index, (input_ids, example) in enumerate(zip(all_input_ids, self
        .examples)):
        if ex_index % 10000 == 0:
            logger.info('Writing example %d/%d' % (ex_index, len(self.
                examples)))
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = batch_length - len(input_ids)
        if pad_on_left:
            input_ids = [pad_token] * padding_length + input_ids
            attention_mask = [0 if mask_padding_with_zero else 1
                ] * padding_length + attention_mask
        else:
            input_ids = input_ids + [pad_token] * padding_length
            attention_mask = attention_mask + [0 if mask_padding_with_zero else
                1] * padding_length
        assert len(input_ids
            ) == batch_length, 'Error with input length {} vs {}'.format(len
            (input_ids), batch_length)
        assert len(attention_mask
            ) == batch_length, 'Error with input length {} vs {}'.format(len
            (attention_mask), batch_length)
        if self.mode == 'classification':
            label = label_map[example.label]
        elif self.mode == 'regression':
            label = float(example.label)
        else:
            raise ValueError(self.mode)
        if ex_index < 5 and self.verbose:
            logger.info('*** Example ***')
            logger.info('guid: %s' % example.guid)
            logger.info('input_ids: %s' % ' '.join([str(x) for x in input_ids])
                )
            logger.info('attention_mask: %s' % ' '.join([str(x) for x in
                attention_mask]))
            logger.info('label: %s (id = %d)' % (example.label, label))
        features.append(InputFeatures(input_ids=input_ids, attention_mask=
            attention_mask, label=label))
    if return_tensors is None:
        return features
    elif return_tensors == 'tf':
        if not is_tf_available():
            raise RuntimeError(
                "return_tensors set to 'tf' but TensorFlow 2.0 can't be imported"
                )
        import tensorflow as tf

        def gen():
            for ex in features:
                yield {'input_ids': ex.input_ids, 'attention_mask': ex.
                    attention_mask}, ex.label
        dataset = tf.data.Dataset.from_generator(gen, ({'input_ids': tf.
            int32, 'attention_mask': tf.int32}, tf.int64), ({'input_ids':
            tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]
            )}, tf.TensorShape([])))
        return dataset
    elif return_tensors == 'pt':
        if not is_torch_available():
            raise RuntimeError(
                "return_tensors set to 'pt' but PyTorch can't be imported")
        import torch
        from torch.utils.data import TensorDataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype
            =torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in
            features], dtype=torch.long)
        if self.mode == 'classification':
            all_labels = torch.tensor([f.label for f in features], dtype=
                torch.long)
        elif self.mode == 'regression':
            all_labels = torch.tensor([f.label for f in features], dtype=
                torch.float)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
        return dataset
    else:
        raise ValueError("return_tensors should be one of 'tf' or 'pt'")
