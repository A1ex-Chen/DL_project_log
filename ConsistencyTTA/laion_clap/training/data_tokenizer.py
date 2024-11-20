def tokenizer(text, tmodel='roberta', max_length=77):
    """tokenizer for different models
    tmodel is default to roberta as it is the best model for our task
    max_length is default to 77 from the OpenAI CLIP parameters
    We assume text to be a single string, but it can also be a list of strings
    """
    if tmodel == 'transformer':
        return clip_tokenizer(text).squeeze(0)
    elif tmodel == 'bert':
        result = bert_tokenizer(text, padding='max_length', truncation=True,
            max_length=max_length, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in result.items()}
    elif tmodel == 'roberta':
        result = roberta_tokenizer(text, padding='max_length', truncation=
            True, max_length=max_length, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in result.items()}
    elif tmodel == 'bart':
        result = bart_tokenizer(text, padding='max_length', truncation=True,
            max_length=max_length, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in result.items()}
