def update_data_paths(self, data_dict):
    path = data_dict.get('path', '')
    for split in ['train', 'val', 'test']:
        if data_dict.get(split):
            split_path = data_dict.get(split)
            data_dict[split] = f'{path}/{split_path}' if isinstance(split, str
                ) else [f'{path}/{x}' for x in split_path]
    return data_dict
