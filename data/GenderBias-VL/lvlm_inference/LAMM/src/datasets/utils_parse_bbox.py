def parse_bbox(text):
    num_list = []
    num_list = parse_num(num_list, '[', ']', text)
    num_list = parse_num(num_list, '(', ')', text)
    bbox_list = []
    num_list = num_list[:len(num_list) // 4 * 4]
    if len(num_list) == 0:
        str_list = num_pattern.findall(text)
        num_list = [float(item) for item in str_list]
        num_list = num_list[:len(num_list) // 4 * 4]
    for i in range(0, len(num_list), 4):
        cur_bbox = [num_list[j] for j in range(i, i + 4)]
        if cur_bbox[0] > cur_bbox[2] and cur_bbox[1] > cur_bbox[3]:
            cur_bbox = [cur_bbox[2], cur_bbox[3], cur_bbox[0], cur_bbox[1]]
        if cur_bbox[0] <= cur_bbox[2] and cur_bbox[1] <= cur_bbox[3]:
            bbox_list.append(cur_bbox)
    return bbox_list
