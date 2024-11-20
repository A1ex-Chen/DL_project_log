def get_preprocess_function(text_column, outcome_column, tokenizer,
    mask_token, max_length):

    def preprocess_function(examples):
        prompt_template = '{} ' + mask_token + ' {}'
        inputs = examples[text_column]
        targets = examples[outcome_column]
        input_list = [prompt_template.format(i, t) for i, t in zip(inputs,
            targets)]
        model_inputs = tokenizer(input_list, max_length=max_length, padding
            =False, truncation=True)
        return model_inputs
    return preprocess_function
