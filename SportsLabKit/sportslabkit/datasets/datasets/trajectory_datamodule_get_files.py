def get_files(self, data_dir):
    files = []
    for file in data_dir.glob('*.txt'):
        files.append(file)
    return files
