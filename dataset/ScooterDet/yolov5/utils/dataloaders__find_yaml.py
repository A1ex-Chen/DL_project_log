@staticmethod
def _find_yaml(dir):
    files = list(dir.glob('*.yaml')) or list(dir.rglob('*.yaml'))
    assert files, f'No *.yaml file found in {dir}'
    if len(files) > 1:
        files = [f for f in files if f.stem == dir.stem]
        assert files, f'Multiple *.yaml files found in {dir}, only 1 *.yaml file allowed'
    assert len(files
        ) == 1, f'Multiple *.yaml files found: {files}, only 1 *.yaml file allowed in {dir}'
    return files[0]
