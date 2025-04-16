def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = '### '
    END_SIGNAL = '\n'
    conversation = header
    for sentence in source:
        from_str = sentence['from']
        if from_str.lower() == 'human':
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == 'gpt':
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence['value'] = BEGIN_SIGNAL + from_str + ': ' + sentence['value'
            ] + END_SIGNAL
        if get_conversation:
            conversation += sentence['value']
    conversation += BEGIN_SIGNAL
    return conversation
