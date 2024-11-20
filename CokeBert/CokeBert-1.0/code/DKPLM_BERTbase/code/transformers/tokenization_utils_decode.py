def decode(self, token_ids, skip_special_tokens=False,
    clean_up_tokenization_spaces=True):
    """
        Converts a sequence of ids (integer) in a string, using the tokenizer and vocabulary
        with options to remove special tokens and clean up tokenization spaces.
        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.

        Args:
            token_ids: list of tokenized input ids. Can be obtained using the `encode` or `encode_plus` methods.
            skip_special_tokens: if set to True, will replace special tokens.
            clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.
        """
    filtered_tokens = self.convert_ids_to_tokens(token_ids,
        skip_special_tokens=skip_special_tokens)
    sub_texts = []
    current_sub_text = []
    for token in filtered_tokens:
        if skip_special_tokens and token in self.all_special_ids:
            continue
        if token in self.added_tokens_encoder:
            if current_sub_text:
                sub_texts.append(self.convert_tokens_to_string(
                    current_sub_text))
                current_sub_text = []
            sub_texts.append(' ' + token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_texts.append(self.convert_tokens_to_string(current_sub_text))
    text = ''.join(sub_texts)
    if clean_up_tokenization_spaces:
        clean_text = self.clean_up_tokenization(text)
        return clean_text
    else:
        return text
