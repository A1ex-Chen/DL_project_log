def preprocess(data, input_dir, dest_dir, target_sr=None, speed=None,
    overwrite=True):
    speed = speed or []
    speed.append(1)
    speed = list(set(speed))
    input_fname = os.path.join(input_dir, data['input_relpath'], data[
        'input_fname'])
    input_sr = sox.file_info.sample_rate(input_fname)
    target_sr = target_sr or input_sr
    os.makedirs(os.path.join(dest_dir, data['input_relpath']), exist_ok=True)
    output_dict = {}
    output_dict['transcript'] = data['transcript'].lower().strip()
    output_dict['files'] = []
    fname = os.path.splitext(data['input_fname'])[0]
    for s in speed:
        output_fname = fname + '{}.wav'.format('' if s == 1 else '-{}'.
            format(s))
        output_fpath = os.path.join(dest_dir, data['input_relpath'],
            output_fname)
        if not os.path.exists(output_fpath) or overwrite:
            cbn = sox.Transformer().speed(factor=s).convert(target_sr)
            cbn.build(input_fname, output_fpath)
        file_info = sox.file_info.info(output_fpath)
        file_info['fname'] = os.path.join(os.path.basename(dest_dir), data[
            'input_relpath'], output_fname)
        file_info['speed'] = s
        output_dict['files'].append(file_info)
        if s == 1:
            file_info = sox.file_info.info(output_fpath)
            output_dict['original_duration'] = file_info['duration']
            output_dict['original_num_samples'] = file_info['num_samples']
    return output_dict
