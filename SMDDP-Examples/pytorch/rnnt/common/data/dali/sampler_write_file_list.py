def write_file_list(self, names, labels):
    with open(self.file_list_path, 'w') as f:
        f.writelines(f'{name} {label}\n' for name, label in zip(names, labels))
