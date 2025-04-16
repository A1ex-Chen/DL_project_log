def get_dataset_size(self):
    assert self.dataset_size, 'Dataset size not known. Run make_file_list first'
    return self.dataset_size
