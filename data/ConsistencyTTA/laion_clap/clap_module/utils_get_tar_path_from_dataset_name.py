def get_tar_path_from_dataset_name(dataset_names, dataset_types, islocal,
    dataset_path, proportion=1, full_dataset=None):
    """
    Get tar path from dataset name and type
    """
    output = []
    for n in dataset_names:
        if full_dataset is not None and n in full_dataset:
            current_dataset_types = dataset_split[n]
        else:
            current_dataset_types = dataset_types
        for s in current_dataset_types:
            tmp = []
            if islocal:
                sizefilepath_ = f'{dataset_path}/{n}/{s}/sizes.json'
                if not os.path.exists(sizefilepath_):
                    sizefilepath_ = f'./json_files/{n}/{s}/sizes.json'
            else:
                sizefilepath_ = f'./json_files/{n}/{s}/sizes.json'
            if not os.path.exists(sizefilepath_):
                continue
            sizes = json.load(open(sizefilepath_, 'r'))
            for k in sizes.keys():
                if islocal:
                    tmp.append(f'{dataset_path}/{n}/{s}/{k}')
                else:
                    tmp.append(
                        f'pipe:aws s3 --cli-connect-timeout 0 cp s3://s-laion-audio/webdataset_tar/{n}/{s}/{k} -'
                        )
            if proportion != 1:
                tmp = random.sample(tmp, int(proportion * len(tmp)))
            output.append(tmp)
    return sum(output, [])
