def singleturn_prompt(task_name, assigned_ids=0, defined_prompt=None, **kwargs
    ):
    """
        return prompt: str
    """
    print('Using singleturn prompt...')
    if defined_prompt is not None:
        assert isinstance(defined_prompt, str
            ), f'The defined prompt must be string. '
        print(
            f'Using user defined prompt: {defined_prompt} for task: {task_name}'
            )
        return defined_prompt
    prompt_name = task_name + '_prompts'
    prompt = '{question}'
    if prompt_name in singleturn_prompt_dict:
        prompt = singleturn_prompt_dict[prompt_name][assigned_ids]
        print(f'Using prompt pool prompt: {prompt} for task: {task_name}')
        return prompt
    print(
        f"No prompt defined for task: {task_name}. Make sure you have the key 'question' in dataset for prompt."
        )
    return prompt
