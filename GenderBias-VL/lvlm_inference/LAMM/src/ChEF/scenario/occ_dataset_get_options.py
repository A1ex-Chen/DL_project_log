def get_options(choices, option_content):
    option_list = []
    for idx, answer in enumerate(choices):
        optionstr = OPTION[idx]
        if option_content:
            option_list.append(f'({optionstr}) {answer}')
        else:
            option_list.append(f'({optionstr}')
    return option_list
