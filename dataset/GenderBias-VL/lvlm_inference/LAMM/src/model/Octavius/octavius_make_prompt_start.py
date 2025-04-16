def make_prompt_start(use_system, vision_type: List[str], task_type: List[
    str], template=conversations.default_conversation) ->List[str]:
    PROMPT_START = [
        f"{template.sep} {template.roles[0]}: {VISION_TAGS['sov'][vision_type_i]}"
         for vision_type_i in vision_type]
    if use_system:
        if task_type == 'normal':
            return [(f'{template.system}\n\n' + prompt_start_i) for
                prompt_start_i in PROMPT_START]
        elif template.sys_temp is None:
            return [(f'{conversations.conversation_dict[task]}\n\n' +
                PROMPT_START[i]) for i, task in enumerate(task_type)]
        else:
            return [(template.sys_temp.format(system_message=conversations.
                conversation_dict[task]) + PROMPT_START[i]) for i, task in
                enumerate(task_type)]
    else:
        return PROMPT_START
