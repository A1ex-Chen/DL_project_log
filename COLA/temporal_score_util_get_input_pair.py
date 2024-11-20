def get_input_pair(text_event_list, outcome_event_list):
    input_list = []
    for text in text_event_list:
        text = _remove_punct(text)
        for outcome in outcome_event_list:
            outcome = _sent_lowercase(outcome)
            input_list.append({'text': text, 'outcome': outcome})
    return input_list
