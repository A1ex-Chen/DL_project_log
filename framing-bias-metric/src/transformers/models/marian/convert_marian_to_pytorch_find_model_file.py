def find_model_file(dest_dir):
    model_files = list(Path(dest_dir).glob('*.npz'))
    assert len(model_files) == 1, model_files
    model_file = model_files[0]
    return model_file
