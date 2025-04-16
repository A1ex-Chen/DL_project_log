def remove_prefix(text: str, prefix: str):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
