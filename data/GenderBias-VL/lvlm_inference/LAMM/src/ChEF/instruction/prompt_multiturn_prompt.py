def multiturn_prompt(task_name, assigned_ids=0, defined_prompt=None, **kwargs):
    """
        return [prompt_1: str, prompt_2: str]
    """
    print('Using multiturn prompt...')
    if defined_prompt is not None:
        if isinstance(defined_prompt, str):
            defined_prompt = [defined_prompt, defined_prompt]
        print(
            f'Using user defined prompt: {defined_prompt} for task: {task_name}'
            )
        return defined_prompt
    prompt_name = task_name + '_multiturn_prompts'
    prompt = ['{question}', '{question}']
    if prompt_name in multiturn_prompt_dict:
        prompt = multiturn_prompt_dict[prompt_name][assigned_ids]
        if isinstance(prompt, str):
            prompt = [prompt, prompt]
        print(f'Using prompt pool prompt: {prompt} for task: {task_name}')
        return prompt
    print(
        f"No prompt defined for task: {task_name}. Make sure you have the key 'question' in dataset for prompt."
        )
    return prompt
