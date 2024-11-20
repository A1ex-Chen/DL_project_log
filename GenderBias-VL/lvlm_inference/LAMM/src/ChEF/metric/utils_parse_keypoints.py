def parse_keypoints(text):
    num_list = []
    num_list = parse_num(num_list, '[', ']', text)
    num_list = parse_num(num_list, '(', ')', text)
    keypoints_list = []
    num_list = num_list[:len(num_list) // 2 * 2]
    if len(num_list) == 0:
        str_list = num_pattern.findall(text)
        num_list = [float(item) for item in str_list]
        num_list = num_list[:len(num_list) // 2 * 2]
    for i in range(0, len(num_list), 2):
        cur_kps = [num_list[j] for j in range(i, i + 2)]
        keypoints_list.append(cur_kps)
    return keypoints_list
