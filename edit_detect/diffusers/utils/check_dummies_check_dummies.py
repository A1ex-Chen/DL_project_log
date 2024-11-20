def check_dummies(overwrite=False):
    """Check if the dummy files are up to date and maybe `overwrite` with the right content."""
    dummy_files = create_dummy_files()
    short_names = {'torch': 'pt'}
    path = os.path.join(PATH_TO_DIFFUSERS, 'utils')
    dummy_file_paths = {backend: os.path.join(path,
        f'dummy_{short_names.get(backend, backend)}_objects.py') for
        backend in dummy_files.keys()}
    actual_dummies = {}
    for backend, file_path in dummy_file_paths.items():
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', newline='\n') as f:
                actual_dummies[backend] = f.read()
        else:
            actual_dummies[backend] = ''
    for backend in dummy_files.keys():
        if dummy_files[backend] != actual_dummies[backend]:
            if overwrite:
                print(
                    f'Updating diffusers.utils.dummy_{short_names.get(backend, backend)}_objects.py as the main __init__ has new objects.'
                    )
                with open(dummy_file_paths[backend], 'w', encoding='utf-8',
                    newline='\n') as f:
                    f.write(dummy_files[backend])
            else:
                raise ValueError(
                    f'The main __init__ has objects that are not present in diffusers.utils.dummy_{short_names.get(backend, backend)}_objects.py. Run `make fix-copies` to fix this.'
                    )
