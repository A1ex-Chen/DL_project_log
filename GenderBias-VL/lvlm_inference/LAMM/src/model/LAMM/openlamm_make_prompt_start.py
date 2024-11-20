def make_prompt_start(use_system=False, vision_type='image', task_type=
    'normal', template=conversations.default_conversation):
    """make starting prompt

    :param bool use_system: whether to use system message, defaults to False
    :param str vision_type: type of visio data, defaults to 'image'
    :param str task_type: task type of current sample, defaults to 'normal'
    :return str: resulting starting prompt
    """
    PROMPT_START = (
        f"{template.sep} {template.roles[0]}: {VISION_TAGS['sov'][vision_type]}"
        )
    if use_system:
        if task_type == 'normal':
            return f'{template.system}\n\n' + PROMPT_START
        elif template.sys_temp is None:
            return [(f'{conversations.conversation_dict[task]}\n\n' +
                PROMPT_START) for task in task_type]
        else:
            return [(template.sys_temp.format(system_message=conversations.
                conversation_dict[task]) + PROMPT_START) for task in task_type]
    else:
        return PROMPT_START
