def __call__(self, x):
    try:
        if 'json' in x:
            x_json = json.loads(x['json'])
            filter_size = (x_json.get('original_width', 0.0) or 0.0
                ) >= self.min_size and x_json.get('original_height', 0
                ) >= self.min_size
            filter_watermark = (x_json.get('pwatermark', 1.0) or 1.0
                ) <= self.max_pwatermark
            return filter_size and filter_watermark
        else:
            return False
    except Exception:
        return False
