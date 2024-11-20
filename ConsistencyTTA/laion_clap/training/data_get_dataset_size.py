def get_dataset_size(shards, sizefilepath_=None, is_local=True):
    if isinstance(shards, list):
        size_list = []
        for s in shards:
            size_list.append(get_dataset_size(s, sizefilepath_=
                sizefilepath_, is_local=is_local)[0])
    else:
        if not is_local:
            for n in dataset_split.keys():
                if n in shards.split('/'):
                    break
            for s in dataset_split[n]:
                if s in shards.split('/'):
                    break
            sizefilepath_ = f'./json_files/{n}/{s}/sizes.json'
        shards_list = list(braceexpand.braceexpand(shards))
        dir_path = os.path.dirname(shards)
        if sizefilepath_ is not None:
            sizes = json.load(open(sizefilepath_, 'r'))
            total_size = sum([int(sizes[os.path.basename(shard.replace(
                '.tar -', '.tar'))]) for shard in shards_list])
        else:
            sizes_filename = os.path.join(dir_path, 'sizes.json')
            len_filename = os.path.join(dir_path, '__len__')
            if os.path.exists(sizes_filename):
                sizes = json.load(open(sizes_filename, 'r'))
                total_size = sum([int(sizes[os.path.basename(shard)]) for
                    shard in shards_list])
            elif os.path.exists(len_filename):
                total_size = ast.literal_eval(open(len_filename, 'r').read())
            else:
                raise Exception(
                    f'Cannot find sizes file for dataset {shards}. Please specify the path to the file.'
                    )
        num_shards = len(shards_list)
    if isinstance(shards, list):
        return sum(size_list), len(shards)
    else:
        return total_size, num_shards
