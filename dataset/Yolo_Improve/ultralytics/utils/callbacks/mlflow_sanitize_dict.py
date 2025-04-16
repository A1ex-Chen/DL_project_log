def sanitize_dict(x):
    """Sanitize dictionary keys by removing parentheses and converting values to floats."""
    return {k.replace('(', '').replace(')', ''): float(v) for k, v in x.items()
        }
