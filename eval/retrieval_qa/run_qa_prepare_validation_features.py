def prepare_validation_features(examples):
    examples[question_column_name] = [q.lstrip() for q in examples[
        question_column_name]]
    tokenized_examples = tokenizer(examples[question_column_name if
        pad_on_right else context_column_name], examples[
        context_column_name if pad_on_right else question_column_name],
        truncation='only_second' if pad_on_right else 'only_first',
        max_length=max_seq_length, stride=data_args.doc_stride,
        return_overflowing_tokens=True, return_offsets_mapping=True,
        padding='max_length' if data_args.pad_to_max_length else False)
    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
    tokenized_examples['example_id'] = []
    for i in range(len(tokenized_examples['input_ids'])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
        sample_index = sample_mapping[i]
        tokenized_examples['example_id'].append(examples['id'][sample_index])
        tokenized_examples['offset_mapping'][i] = [(o if sequence_ids[k] ==
            context_index else None) for k, o in enumerate(
            tokenized_examples['offset_mapping'][i])]
    return tokenized_examples
