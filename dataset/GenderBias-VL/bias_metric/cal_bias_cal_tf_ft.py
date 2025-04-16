def cal_tf_ft(tf_data, ft_data):
    tf, ft = 'True->False', 'False->True'
    ans = {'cate_num': tf_data[tf]['cate_num'] + ft_data[ft]['cate_num'],
        'total_num': tf_data[tf]['true_num'] + ft_data[ft]['false_num']}
    return ans
