def _tokenize_str(role, content):
    return f'{role}\n{content}', tokenizer.encode(role, allowed_special=set
        (tokenizer.IMAGE_ST)) + nl_tokens + tokenizer.encode(content,
        allowed_special=set(tokenizer.IMAGE_ST))
