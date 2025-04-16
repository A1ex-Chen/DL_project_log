def get_preprocess_function(data_args, tokenizer, model_name_or_path, joint
    =False):

    def preprocess_function(examples):
        padding = 'max_length' if data_args.pad_to_max_length else False
        inputs = []
        for i in range(len(examples[data_args.text_column])):
            if not joint:
                inputs.append(examples[data_args.text_column][i])
            else:
                last_idx = i - i % 4 + 4
                input_text = ' '.join(examples[data_args.text_column][i:
                    last_idx])
                inputs.append(input_text)
        if not joint:
            prompt = '{} Before that,'
        else:
            prompt = '{} Before all events, '
        if 't5' in model_name_or_path:
            prompt += '<extra_id_0>'
        inputs = [prompt.format(i) for i in inputs]
        model_inputs = [tokenizer(ipt, padding=padding, truncation=True,
            return_tensors='pt') for ipt in inputs]
        return model_inputs
    return preprocess_function
