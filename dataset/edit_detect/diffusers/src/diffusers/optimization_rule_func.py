def rule_func(steps: int) ->float:
    sorted_steps = sorted(rules_dict.keys())
    for i, sorted_step in enumerate(sorted_steps):
        if steps < sorted_step:
            return rules_dict[sorted_steps[i]]
    return last_lr_multiple
