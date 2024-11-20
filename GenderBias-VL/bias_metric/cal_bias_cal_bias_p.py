def cal_bias_p(P1, P2):
    cate_num = P1['cate_num'] - P2['cate_num']
    total_num = P1['total_num'] + P2['total_num']
    return cate_num, total_num
