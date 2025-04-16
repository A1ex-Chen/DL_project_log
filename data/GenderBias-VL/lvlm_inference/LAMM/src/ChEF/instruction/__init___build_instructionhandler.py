def build_instructionhandler(task_name, dataset, prompt_type='singleturn',
    prompt_assigned_ids=0, template_assigned_ids=0, incontext_cfg=None, **
    kwargs):
    assert prompt_type in supported_prompt_types, f'Supported prompt types are {supported_prompt_types}, got {prompt_type}'
    prompt = build_prompt(task_name=task_name, prompt_type=prompt_type,
        assigned_ids=prompt_assigned_ids, **kwargs)
    template = build_template(task_name=task_name, assigned_ids=
        template_assigned_ids, prompt_type=prompt_type, **kwargs)
    handler = InstructionHandler(prompt, template, incontext_cfg=
        incontext_cfg, dataset=dataset)
    return handler
