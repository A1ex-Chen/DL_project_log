def split_on_token(tok, text):
    result = []
    tok_extended = all_special_tokens_extended.get(tok, None)
    split_text = text.split(tok)
    full_word = ''
    for i, sub_text in enumerate(split_text):
        if isinstance(tok_extended, AddedToken):
            if tok_extended.single_word:
                if i < len(split_text) - 1 and not _is_end_of_word(sub_text
                    ) and not _is_start_of_word(split_text[i + 1]):
                    full_word += sub_text + tok
                elif full_word:
                    full_word += sub_text
                    result.append(full_word)
                    full_word = ''
                    continue
            if tok_extended.rstrip and i > 0:
                sub_text = sub_text.lstrip()
            if tok_extended.lstrip and i < len(split_text) - 1:
                sub_text = sub_text.rstrip()
        else:
            if i < len(split_text) - 1:
                sub_text = sub_text.rstrip()
            if i > 0:
                sub_text = sub_text.lstrip()
        if i == 0 and not sub_text:
            result.append(tok)
        elif i == len(split_text) - 1:
            if sub_text:
                result.append(sub_text)
            else:
                pass
        else:
            if sub_text:
                result.append(sub_text)
            result.append(tok)
    return result
