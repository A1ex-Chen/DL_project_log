def parse_key_value_pair(pair):
    """Parse one 'key=value' pair and return key and value."""
    k, v = pair.split('=', 1)
    k, v = k.strip(), v.strip()
    assert v, f"missing '{k}' value"
    return k, smart_value(v)
