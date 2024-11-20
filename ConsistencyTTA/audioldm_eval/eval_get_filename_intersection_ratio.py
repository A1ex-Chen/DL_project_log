def get_filename_intersection_ratio(self, dir1, dir2, threshold=0.99,
    limit_num=None):
    self.datalist1 = [os.path.join(dir1, x) for x in os.listdir(dir1)]
    self.datalist1 = sorted(self.datalist1)
    self.datalist1 = [item for item in self.datalist1 if item.endswith('.wav')]
    self.datalist2 = [os.path.join(dir2, x) for x in os.listdir(dir2)]
    self.datalist2 = sorted(self.datalist2)
    self.datalist2 = [item for item in self.datalist2 if item.endswith('.wav')]
    data_dict1 = {os.path.basename(x): x for x in self.datalist1}
    data_dict2 = {os.path.basename(x): x for x in self.datalist2}
    keyset1 = set(data_dict1.keys())
    keyset2 = set(data_dict2.keys())
    intersect_keys = keyset1.intersection(keyset2)
    if len(intersect_keys) / len(keyset1) > threshold and len(intersect_keys
        ) / len(keyset2) > threshold:
        return True
    else:
        return False
