def eval_cost(response_list):
    inputs = 0
    outputs = 0
    cost = 0.0
    for item in response_list:
        ic = item['usage']['prompt_tokens']
        oc = item['usage']['completion_tokens']
        inputs += ic
        outputs += oc
    print(inputs)
    print(outputs)
    cost = inputs / 1000 * 0.0015 + outputs / 1000 * 0.002
    return cost
