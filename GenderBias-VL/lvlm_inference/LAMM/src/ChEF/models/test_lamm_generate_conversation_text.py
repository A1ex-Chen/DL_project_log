def generate_conversation_text(self, input_list, history=[], sys_msg=None):
    """get all conversation text

        :param args args: input args
        :param str question: current input from user
        :param list history: history of conversation, [(q, a)]
        """
    conv = conv_templates[self.conv_mode]
    if sys_msg:
        conv.system = sys_msg
    prompts_list = []
    for input in input_list:
        prompts = ''
        prompts += conv.system
        for q, a in history:
            prompts += '{} {}: {}\n{} {}: {}\n'.format(conv.sep, conv.roles
                [0], q, conv.sep, conv.roles[1], a)
        prompts += '{} {}: {}\n'.format(conv.sep, conv.roles[0], input)
        prompts_list.append(prompts)
    return prompts_list
