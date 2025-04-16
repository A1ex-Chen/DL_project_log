def remove_suffix(text: str, suffix: str):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text
