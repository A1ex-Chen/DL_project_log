def list_and_dict_from_file(self, filepath):
    with open(filepath, 'r') as f:
        file_list = f.read().splitlines()
    dictionary = dict()
    for cam in self.cameras:
        dictionary[cam] = [i for i in file_list if cam in i]
    return file_list, dictionary
