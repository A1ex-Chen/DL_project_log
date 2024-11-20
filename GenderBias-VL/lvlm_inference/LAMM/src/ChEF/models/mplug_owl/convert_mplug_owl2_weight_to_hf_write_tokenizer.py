def write_tokenizer(tokenizer_path, input_tokenizer_path):
    tokenizer_class = (LlamaTokenizer if LlamaTokenizerFast is None else
        LlamaTokenizerFast)
    print(f'Saving a {tokenizer_class.__name__} to {tokenizer_path}.')
    tokenizer = tokenizer_class(input_tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
