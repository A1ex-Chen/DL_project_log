def adjust_phrase_positions(phrases, text):
    positions = []
    for phrase, start, end in phrases:
        adjusted_start = len(remove_special_fields(text[:start]))
        adjusted_end = len(remove_special_fields(text[:end]))
        positions.append((phrase, adjusted_start, adjusted_end))
    return positions
