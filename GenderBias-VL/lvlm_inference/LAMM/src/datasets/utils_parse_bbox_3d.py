def parse_bbox_3d(text):
    num_list = []
    num_list = parse_num(num_list, '[', ']', text)
    num_list = parse_num(num_list, '(', ')', text)
    bbox_list = []
    num_list = num_list[:len(num_list) // 6 * 6]
    if len(num_list) == 0:
        str_list = num_pattern.findall(text)
        num_list = [float(item) for item in str_list]
        num_list = num_list[:len(num_list) // 6 * 6]
    for i in range(0, len(num_list), 6):
        cur_bbox = [num_list[j] for j in range(i, i + 6)]
        bbox_list.append(cur_bbox)
    return bbox_list
