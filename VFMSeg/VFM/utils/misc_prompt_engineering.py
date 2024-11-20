def prompt_engineering(classnames, topk=1, suffix='.'):
    prompt_templates = get_prompt_templates()
    temp_idx = np.random.randint(min(len(prompt_templates), topk))
    if isinstance(classnames, list):
        classname = random.choice(classnames)
    else:
        classname = classnames
    return prompt_templates[temp_idx].replace('.', suffix).format(classname
        .replace(',', '').replace('+', ' '))
