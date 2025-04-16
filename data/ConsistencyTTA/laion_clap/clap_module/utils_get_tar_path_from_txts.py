def get_tar_path_from_txts(txt_path, islocal, proportion=1):
    """
    Get tar path from txt path
    """
    if isinstance(txt_path, (list, tuple)):
        return sum([get_tar_path_from_txts(txt_path[i], islocal=islocal,
            proportion=proportion) for i in range(len(txt_path))], [])
    if isinstance(txt_path, str):
        with open(txt_path) as f:
            lines = f.readlines()
        if islocal:
            lines = [lines[i].split('\n')[0].replace(
                'pipe:aws s3 cp s3://s-laion-audio/', '/mnt/audio_clip/') for
                i in range(len(lines))]
        else:
            lines = [lines[i].split('\n')[0].replace('.tar', '.tar -') for
                i in range(len(lines))]
        if proportion != 1:
            print('Sampling tars with proportion of {}'.format(proportion))
            lines = random.sample(lines, int(proportion * len(lines)))
        return lines
