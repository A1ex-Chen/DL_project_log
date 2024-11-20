def get_label(self, data_dir):
    """See base class."""
    filename = os.path.join(data_dir, 'label.dict')
    v = []
    with open(filename, 'r') as fin:
        for line in fin:
            vec = line.split('\t')
            v.append(vec[1])
    return v
