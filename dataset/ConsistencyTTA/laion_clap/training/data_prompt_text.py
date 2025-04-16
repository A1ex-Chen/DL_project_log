def prompt_text(self, target):
    events = _AUDIOSET_MAP[np.where(target > 0)]
    event_text = 'The sounds of ' + ', '.join(events[:-1]) + ' and ' + events[
        -1]
    text = tokenizer(event_text)[0]
    return text
