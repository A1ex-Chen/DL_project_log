def get_files(self):
    """Returns a list of paths for all relevant model files given by the user."""
    files = []
    for path in self.paths:
        path = Path(path)
        if path.is_dir():
            extensions = ['*.pt', '*.onnx', '*.yaml']
            files.extend([file for ext in extensions for file in glob.glob(
                str(path / ext))])
        elif path.suffix in {'.pt', '.yaml', '.yml'}:
            files.append(str(path))
        else:
            files.extend(glob.glob(str(path)))
    print(f'Profiling: {sorted(files)}')
    return [Path(file) for file in sorted(files)]
