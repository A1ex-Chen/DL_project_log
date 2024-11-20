def _save_dataset_yaml(self):
    yaml_path = os.path.join(self._save_path_pfx, 'dataset.yaml')
    with open(yaml_path, 'w+') as yaml_file:
        yaml_file.write('train: %s\n' % os.path.join(self._image_dir_path,
            'train/'))
        yaml_file.write('val: %s\n\n' % os.path.join(self._image_dir_path,
            'val/'))
        yaml_file.write('nc: %i\n\n' % len(self._label_id_map))
        names_str = ''
        for label, _ in self._label_id_map.items():
            names_str += "'%s', " % label
        names_str = names_str.rstrip(', ')
        yaml_file.write('names: [%s]' % names_str)
