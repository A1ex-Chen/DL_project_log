def _update_label_text(self, labels):
    """Update label text."""
    if 'texts' not in labels:
        return labels
    mix_texts = sum([labels['texts']] + [x['texts'] for x in labels[
        'mix_labels']], [])
    mix_texts = list({tuple(x) for x in mix_texts})
    text2id = {text: i for i, text in enumerate(mix_texts)}
    for label in ([labels] + labels['mix_labels']):
        for i, cls in enumerate(label['cls'].squeeze(-1).tolist()):
            text = label['texts'][int(cls)]
            label['cls'][i] = text2id[tuple(text)]
        label['texts'] = mix_texts
    return labels
