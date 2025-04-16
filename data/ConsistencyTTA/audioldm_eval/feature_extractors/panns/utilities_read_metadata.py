def read_metadata(csv_path, classes_num, id_to_ix):
    """Read metadata of AudioSet from a csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict: {'audio_name': (audios_num,), 'target': (audios_num, classes_num)}
    """
    with open(csv_path, 'r') as fr:
        lines = fr.readlines()
        lines = lines[3:]
    audios_num = len(lines)
    targets = np.zeros((audios_num, classes_num), dtype=np.bool)
    audio_names = []
    for n, line in enumerate(lines):
        items = line.split(', ')
        """items: ['--4gqARaEJE', '0.000', '10.000', '"/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"
']"""
        audio_name = 'Y{}.wav'.format(items[0])
        label_ids = items[3].split('"')[1].split(',')
        audio_names.append(audio_name)
        for id in label_ids:
            ix = id_to_ix[id]
            targets[n, ix] = 1
    meta_dict = {'audio_name': np.array(audio_names), 'target': targets}
    return meta_dict
