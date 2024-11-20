def check_option(res_list, gt_char):
    for res in res_list:
        if gt_char not in res:
            return False
    return True
