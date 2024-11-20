def merge_adjacent_words(marked_words):
    merged_words = []
    current_word, current_flag = marked_words[0]
    for word, flag in marked_words[1:]:
        if flag == current_flag:
            current_word += ' ' + word
        else:
            merged_words.append((current_word, current_flag))
            current_word = word
            current_flag = flag
    merged_words.append((current_word, current_flag))
    return merged_words
