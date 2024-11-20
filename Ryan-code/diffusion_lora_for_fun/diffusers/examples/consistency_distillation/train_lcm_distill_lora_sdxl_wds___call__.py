def __call__(self, x):
    try:
        if 'json' in x:
            x_json = json.loads(x['json'])
            filter_size = (x_json.get(WDS_JSON_WIDTH, 0.0) or 0.0
                ) >= self.min_size and x_json.get(WDS_JSON_HEIGHT, 0
                ) >= self.min_size
            filter_watermark = (x_json.get('pwatermark', 0.0) or 0.0
                ) <= self.max_pwatermark
            return filter_size and filter_watermark
        else:
            return False
    except Exception:
        return False
