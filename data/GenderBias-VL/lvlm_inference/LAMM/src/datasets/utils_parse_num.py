def parse_num(num_list, split_char_a, split_char_b, text):
    flag = 0
    tmpnum = ''
    for c in text:
        if c == split_char_a:
            flag = 1
        elif c == split_char_b:
            flag = 0
            if is_number(tmpnum):
                num_list.append(float(tmpnum))
                tmpnum = ''
        elif flag == 0:
            continue
        elif c != ',' and c != ' ':
            tmpnum += c
        elif is_number(tmpnum):
            num_list.append(float(tmpnum))
            tmpnum = ''
    return num_list
