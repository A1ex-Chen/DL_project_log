def sample_prop(sizefile, inputs, proportion, is_local=True):
    """
    Sample a proportion of the data.
    """
    file_path_dict = {os.path.split(inputs[i])[1]: os.path.split(inputs[i])
        [0] for i in range(len(inputs))}
    sampled_filepath_dict = {}
    sampled_size_dict = {}
    if not is_local:
        if os.path.exists('sizes.json'):
            os.remove('sizes.json')
        wget.download(sizefile, 'sizes.json')
        sizefile = 'sizes.json'
    with open(sizefile, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)
    L = int(len(file_path_dict) * proportion)
    subkeys = random.sample(file_path_dict.keys(), L)
    for k in subkeys:
        sampled_size_dict[k] = load_dict[k]
        sampled_filepath_dict[k] = file_path_dict[k]
    return sum(sampled_size_dict.values()), L, [os.path.join(v, k) for k, v in
        sampled_filepath_dict.items()], sampled_size_dict
