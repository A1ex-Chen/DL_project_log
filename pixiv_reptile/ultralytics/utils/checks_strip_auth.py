def strip_auth(v):
    """Clean longer Ultralytics HUB URLs by stripping potential authentication information."""
    return clean_url(v) if isinstance(v, str) and v.startswith('http') and len(
        v) > 100 else v
