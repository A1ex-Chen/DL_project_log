def read_all_labels(ann_root):
    ann_list = os.listdir(ann_root)
    all_labels = {}
    for ann_file in ann_list:
        if not ann_file.endswith('txt'):
            continue
        ann_labels = read_label_txt(os.path.join(ann_root, ann_file))
        all_labels[ann_file] = ann_labels
    return all_labels
