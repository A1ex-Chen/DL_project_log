def align_two_file_list(self):
    data_dict1 = {os.path.basename(x): x for x in self.datalist1}
    data_dict2 = {os.path.basename(x): x for x in self.datalist2}
    keyset1 = set(data_dict1.keys())
    keyset2 = set(data_dict2.keys())
    intersect_keys = keyset1.intersection(keyset2)
    self.datalist1 = [data_dict1[k] for k in intersect_keys]
    self.datalist2 = [data_dict2[k] for k in intersect_keys]
