def split_on_token(tok, text):
    result = []
    split_text = text.split(tok)
    for i, sub_text in enumerate(split_text):
        sub_text = sub_text.strip()
        if i == 0 and not sub_text:
            result += [tok]
        elif i == len(split_text) - 1:
            if sub_text:
                result += [sub_text]
            else:
                pass
        else:
            if sub_text:
                result += [sub_text]
            result += [tok]
    return result
