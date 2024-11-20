def _replace_html_entities(text, keep=(), remove_illegal=True, encoding='utf-8'
    ):
    """
    Remove entities from text by converting them to their corresponding unicode character.

    Args:
        text:
            A unicode string or a byte string encoded in the given `encoding` (which defaults to 'utf-8').
        keep (list):
            List of entity names which should not be replaced. This supports both numeric entities (``&#nnnn;`` and
            ``&#hhhh;``) and named entities (such as ``&nbsp;`` or ``&gt;``).
        remove_illegal (bool):
            If `True`, entities that can't be converted are removed. Otherwise, entities that can't be converted are
            kept "as is".

    Returns: A unicode string with the entities removed.

    See https://github.com/scrapy/w3lib/blob/master/w3lib/html.py

        >>> from nltk.tokenize.casual import _replace_html_entities >>> _replace_html_entities(b'Price: &pound;100')
        'Price: \\xa3100' >>> print(_replace_html_entities(b'Price: &pound;100')) Price: £100 >>>
    """

    def _convert_entity(match):
        entity_body = match.group(3)
        if match.group(1):
            try:
                if match.group(2):
                    number = int(entity_body, 16)
                else:
                    number = int(entity_body, 10)
                if 128 <= number <= 159:
                    return bytes((number,)).decode('cp1252')
            except ValueError:
                number = None
        elif entity_body in keep:
            return match.group(0)
        else:
            number = html.entities.name2codepoint.get(entity_body)
        if number is not None:
            try:
                return chr(number)
            except (ValueError, OverflowError):
                pass
        return '' if remove_illegal else match.group(0)
    return ENT_RE.sub(_convert_entity, _str_to_unicode(text, encoding))
