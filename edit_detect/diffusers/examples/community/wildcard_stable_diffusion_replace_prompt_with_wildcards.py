def replace_prompt_with_wildcards(prompt: str, wildcard_option_dict: Dict[
    str, List[str]]={}, wildcard_files: List[str]=[]):
    new_prompt = prompt
    wildcard_option_dict = grab_wildcard_values(wildcard_option_dict,
        wildcard_files)
    for m in global_re_wildcard.finditer(new_prompt):
        wildcard_value = m.group()
        replace_value = random.choice(wildcard_option_dict[wildcard_value.
            strip('__')])
        new_prompt = new_prompt.replace(wildcard_value, replace_value, 1)
    return new_prompt
