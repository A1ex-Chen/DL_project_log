def english_cleaners(text, table=None):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    if table is not None:
        text = remove_punctuation(text, table)
    text = collapse_whitespace(text)
    return text
