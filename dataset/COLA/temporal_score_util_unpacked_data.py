def unpacked_data(example_list, text_column, outcome_column):
    event_pair_list = []
    for cur_example in example_list:
        text_event_list = cur_example[text_column]
        outcome_event_list = cur_example[outcome_column]
        event_pair_list.extend(get_input_pair(text_event_list,
            outcome_event_list))
    return event_pair_list
