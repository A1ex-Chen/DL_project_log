def get_acc(res_dict):
    acc_keys = ['ACC', 'vanilla_acc']
    for acc_key in acc_keys:
        if acc_key in res_dict:
            origin_acc = res_dict[acc_key]
            return origin_acc
    return 0
