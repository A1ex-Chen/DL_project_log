def check_text(text, choices, gt_id):
    text = text.lower()
    if choices[gt_id].lower() not in text:
        return False
    for id, choice in enumerate(choices):
        if id == gt_id:
            continue
        if choice.lower() in text:
            return False
    return True
