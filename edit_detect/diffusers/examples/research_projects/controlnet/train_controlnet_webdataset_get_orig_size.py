def get_orig_size(json):
    return int(json.get('original_width', 0.0)), int(json.get(
        'original_height', 0.0))
