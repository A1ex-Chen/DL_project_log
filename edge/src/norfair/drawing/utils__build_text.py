def _build_text(drawable: 'Drawable', draw_labels, draw_ids, draw_scores):
    text = ''
    if draw_labels and drawable.label is not None:
        text = str(drawable.label)
    if draw_ids and drawable.id is not None:
        if len(text) > 0:
            text += '-'
        text += str(drawable.id)
    if draw_scores and drawable.scores is not None:
        if len(text) > 0:
            text += '-'
        text += str(np.round(np.mean(drawable.scores), 4))
    return text
