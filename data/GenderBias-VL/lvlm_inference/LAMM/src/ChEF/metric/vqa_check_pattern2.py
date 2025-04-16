def check_pattern2(res_list, gt_char):
    pred = res_list[0][-1]
    if pred == gt_char:
        return True
    return False
