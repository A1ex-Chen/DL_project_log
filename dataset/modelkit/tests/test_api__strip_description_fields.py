def _strip_description_fields(spec):
    if isinstance(spec, str):
        return '\n'.join(line for line in spec.split('\n') if not any(x in
            line for x in EXCLUDED))
    if isinstance(spec, list):
        return [_strip_description_fields(x) for x in spec]
    if isinstance(spec, dict):
        return {key: _strip_description_fields(value) for key, value in
            spec.items()}
    return spec
