def is_safetensors_compatible(filenames, variant=None, passed_components=None
    ) ->bool:
    """
    Checking for safetensors compatibility:
    - By default, all models are saved with the default pytorch serialization, so we use the list of default pytorch
      files to know which safetensors files are needed.
    - The model is safetensors compatible only if there is a matching safetensors file for every default pytorch file.

    Converting default pytorch serialized filenames to safetensors serialized filenames:
    - For models from the diffusers library, just replace the ".bin" extension with ".safetensors"
    - For models from the transformers library, the filename changes from "pytorch_model" to "model", and the ".bin"
      extension is replaced with ".safetensors"
    """
    pt_filenames = []
    sf_filenames = set()
    passed_components = passed_components or []
    for filename in filenames:
        _, extension = os.path.splitext(filename)
        if len(filename.split('/')) == 2 and filename.split('/')[0
            ] in passed_components:
            continue
        if extension == '.bin':
            pt_filenames.append(os.path.normpath(filename))
        elif extension == '.safetensors':
            sf_filenames.add(os.path.normpath(filename))
    for filename in pt_filenames:
        path, filename = os.path.split(filename)
        filename, extension = os.path.splitext(filename)
        if filename.startswith('pytorch_model'):
            filename = filename.replace('pytorch_model', 'model')
        else:
            filename = filename
        expected_sf_filename = os.path.normpath(os.path.join(path, filename))
        expected_sf_filename = f'{expected_sf_filename}.safetensors'
        if expected_sf_filename not in sf_filenames:
            logger.warning(f'{expected_sf_filename} not found')
            return False
    return True
