def _fetch_latest_picture(self, response, history):
    if history is None:
        history = []
    _history = history + [(response, None)]
    for q, r in _history[::-1]:
        for ele in self.to_list_format(q)[::-1]:
            if 'image' in ele:
                return ele['image']
    return None
