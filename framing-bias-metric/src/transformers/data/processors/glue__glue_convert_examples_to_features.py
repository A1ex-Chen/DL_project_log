def _glue_convert_examples_to_features(examples: List[InputExample],
    tokenizer: PreTrainedTokenizer, max_length: Optional[int]=None, task=
    None, label_list=None, output_mode=None):
    if max_length is None:
        max_length = tokenizer.max_len
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

    def label_from_example(example: InputExample) ->Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == 'classification':
            return label_map[example.label]
        elif output_mode == 'regression':
            return float(example.label)
        raise KeyError(output_mode)
    labels = [label_from_example(example) for example in examples]
    batch_encoding = tokenizer([(example.text_a, example.text_b) for
        example in examples], max_length=max_length, padding='max_length',
        truncation=True)
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)
    for i, example in enumerate(examples[:5]):
        logger.info('*** Example ***')
        logger.info('guid: %s' % example.guid)
        logger.info('features: %s' % features[i])
    return features
