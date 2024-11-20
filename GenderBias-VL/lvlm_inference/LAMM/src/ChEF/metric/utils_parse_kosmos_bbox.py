def parse_kosmos_bbox(text):

    def extract_strings_between_tags(string):
        pattern = '<object>(.*?)</object>'
        matches = re.findall(pattern, string)
        return matches

    def extract_numbers_from_patch(string):
        pattern = '<patch_index_(\\d+)>'
        matches = re.findall(pattern, string)
        return matches

    def index_to_normalized_coordinate(index):
        row = index // 32
        col = index % 32
        normalized_y = row / 32
        normalized_x = col / 32
        return normalized_x, normalized_y
    matches = extract_strings_between_tags(text)
    num_list = []
    for match in matches:
        index_list = extract_numbers_from_patch(match)
        index_list = index_list[:len(index_list) // 2 * 2]
        index_list = [int(index) for index in index_list]
        for index in index_list:
            x, y = index_to_normalized_coordinate(index)
            num_list += [x, y]
    num_list = num_list[:len(num_list) // 4 * 4]
    bbox_list = []
    for i in range(0, len(num_list), 4):
        cur_bbox = [num_list[j] for j in range(i, i + 4)]
        bbox_list.append(cur_bbox)
    return bbox_list
