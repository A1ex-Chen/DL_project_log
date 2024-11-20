def _decode(self, token_ids: List[int], skip_special_tokens: bool=False,
    clean_up_tokenization_spaces: bool=True, spaces_between_special_tokens:
    bool=True) ->str:
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
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_texts.append(self.convert_tokens_to_string(current_sub_text))
    if spaces_between_special_tokens:
        text = ' '.join(sub_texts)
    else:
        text = ''.join(sub_texts)
    if clean_up_tokenization_spaces:
        clean_text = self.clean_up_tokenization(text)
        return clean_text
    else:
        return text
