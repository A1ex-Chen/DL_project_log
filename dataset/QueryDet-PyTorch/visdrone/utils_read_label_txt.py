def read_label_txt(txt_file):
    f = open(txt_file, 'r')
    lines = f.readlines()
    labels = []
    for line in lines:
        line = line.strip().split(',')
        x, y, w, h, not_ignore, cate, trun, occ = line[:8]
        labels.append({'bbox': (int(x), int(y), int(w), int(h)), 'ignore': 
            0 if int(not_ignore) else 1, 'class': int(cate), 'truncate':
            int(trun), 'occlusion': int(occ)})
    return labels
