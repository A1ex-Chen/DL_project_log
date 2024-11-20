def normalize_string(s, charset, punct_map):
    """Normalizes string.

    Example:
        'call me at 8:00 pm!' -> 'call me at eight zero pm'
    """
    charset = set(charset)
    try:
        text = _clean_text(s, ['english_cleaners'], punct_map).strip()
        return ''.join([tok for tok in text if all(t in charset for t in tok)])
    except:
        print(f'WARNING: Normalizing failed: {s}')
        return None
