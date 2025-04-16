def save_p(obj, filename):
    import pickle
    try:
        from deepdiff import DeepDiff
    except:
        os.system('pip install deepdiff')
        from deepdiff import DeepDiff
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(filename, 'rb') as file:
        z = pickle.load(file)
    assert DeepDiff(obj, z, ignore_string_case=True) == {
        }, 'there is something wrong with the saving process'
    return
