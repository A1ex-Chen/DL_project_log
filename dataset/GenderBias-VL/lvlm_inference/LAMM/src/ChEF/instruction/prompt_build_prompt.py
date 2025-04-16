def build_prompt(prompt_type, **kwargs):
    return prompt_func_dict[prompt_type](**kwargs)
