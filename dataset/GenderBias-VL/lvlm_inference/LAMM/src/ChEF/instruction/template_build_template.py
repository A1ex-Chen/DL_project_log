def build_template(task_name, assigned_ids=0, defined_template=None,
    prompt_type='singleturn', **kwargs):
    print('Building answer templates')
    if defined_template is not None:
        template = defined_template
        print(
            f'Using user defined answer template: {template} for task: {task_name}'
            )
    else:
        template_name = task_name + '_templates'
        template = '{option}'
        if template_name in answer_template_dict:
            template = answer_template_dict[template_name][assigned_ids]
            print(
                f'Using answer template pool: {template} for task: {task_name}'
                )
    if prompt_type == 'multiturn':
        if isinstance(template, str):
            template = [template, template]
        print(
            f'Using multiturn answer template: {template} for task: {task_name}'
            )
    return template
