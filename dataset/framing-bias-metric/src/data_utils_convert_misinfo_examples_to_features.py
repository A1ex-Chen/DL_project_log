def convert_misinfo_examples_to_features(examples, tokenizer,
    remove_stopwords, max_length=512, task=None, label_map=None,
    pad_on_left=False, pad_token=0, pad_token_segment_id=0,
    mask_padding_with_zero=True):
    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info('Writing example %d' % ex_index)
        inputs = tokenizer.encode_plus(clean_text(example.text_a,
            remove_stopword=remove_stopwords), clean_text(example.text_b,
            remove_stopword=remove_stopwords) if example.text_b is not None
             else example.text_b, add_special_tokens=True, max_length=
            max_length)
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
        label = label_map[example.label]
        task = example.task
        guid = example.guid
        features.append(InputFeatures(input_ids=input_ids, attention_mask=
            attention_mask, token_type_ids=token_type_ids, label=label,
            task=task, guid=guid))
    return features
