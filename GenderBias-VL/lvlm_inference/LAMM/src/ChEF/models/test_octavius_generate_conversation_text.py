def generate_conversation_text(self, input_list, history, sys_msg=None):
    conv = conv_templates[self.conv_mode]
    if sys_msg:
        if conv.sys_temp is not None:
            conv.system = conv.sys_temp.format(system_message=sys_msg)
        else:
            conv.system = sys_msg
    prompts_list = []
    for input in input_list:
        prompts = ''
        prompts += conv.system + '\n\n'
        for q, a in history:
            prompts += '{}: {}\n{} {}: {}\n{}'.format(conv.roles[0], q,
                conv.sep, conv.roles[1], a, conv.sep2 if conv.sep2 is not
                None else conv.sep)
        prompts += '{}: {}\n{}'.format(conv.roles[0], input, conv.sep)
        prompts_list.append(prompts)
    return prompts_list
