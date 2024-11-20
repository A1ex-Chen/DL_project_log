def mark_texts(text):
    cleaned_text = remove_special_fields(text)
    phrases = find_phrases(text)
    adjusted_phrases = adjust_phrase_positions(phrases, text)
    marked_words = mark_words(cleaned_text, adjusted_phrases)
    merge_words = merge_adjacent_words(marked_words)
    return merge_words
