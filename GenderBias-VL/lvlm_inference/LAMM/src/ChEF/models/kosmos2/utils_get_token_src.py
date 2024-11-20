def get_token_src(self, input, tokenizer, special_tokens=None):
    split_special_token_words = []
    split_results = split_string(input, special_tokens)
    for string in split_results:
        if string in special_tokens:
            split_special_token_words.append(string)
        else:
            encode_tokens = tokenizer.encode(string, out_type=str)
            split_special_token_words.extend(encode_tokens)
    input = ' '.join(split_special_token_words)
    text_tokens = self.source_dictionary.encode_line(input,
        add_if_not_exist=False).tolist()
    text_tokens = text_tokens[:-1]
    return text_tokens
