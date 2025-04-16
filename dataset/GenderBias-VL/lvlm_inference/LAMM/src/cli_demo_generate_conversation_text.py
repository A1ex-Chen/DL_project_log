def generate_conversation_text(args, input, history):
    """get all conversation text

    :param args args: input args
    :param str question: current input from user
    :param list history: history of conversation, [(q, a)]
    """
    assert input is not None or len(input) > 0, 'input is empty!'
    conv = conv_templates[args.conv_mode]
    prompts = ''
    prompts += SYS_MSG
    if len(history) > 0:
        print('{} Q&A found in history...'.format(len(history)))
    for q, a in history:
        prompts += '{} {}: {}\n{} {}: {}\n'.format(conv.sep, conv.roles[0],
            q.replace('<image>', '').replace('\n', ''), conv.sep, conv.
            roles[1], a)
    prompts += '{} {}: {}\n'.format(conv.sep, conv.roles[0], input)
    return prompts
