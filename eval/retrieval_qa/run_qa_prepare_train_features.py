def prepare_train_features(examples):
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
    offset_mapping = tokenized_examples.pop('offset_mapping')
    tokenized_examples['start_positions'] = []
    tokenized_examples['end_positions'] = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples['input_ids'][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]
        if len(answers['answer_start']) == 0:
            tokenized_examples['start_positions'].append(cls_index)
            tokenized_examples['end_positions'].append(cls_index)
        else:
            start_char = answers['answer_start'][0]
            end_char = start_char + len(answers['text'][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0
                ):
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1
            if not (offsets[token_start_index][0] <= start_char and offsets
                [token_end_index][1] >= end_char):
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[
                    token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples['start_positions'].append(
                    token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples['end_positions'].append(token_end_index + 1)
    return tokenized_examples
