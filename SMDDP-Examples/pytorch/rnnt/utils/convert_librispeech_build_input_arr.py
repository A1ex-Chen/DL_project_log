def build_input_arr(input_dir):
    txt_files = glob.glob(os.path.join(input_dir, '**', '*.trans.txt'),
        recursive=True)
    input_data = []
    for txt_file in txt_files:
        rel_path = os.path.relpath(txt_file, input_dir)
        with open(txt_file) as fp:
            for line in fp:
                fname, _, transcript = line.partition(' ')
                input_data.append(dict(input_relpath=os.path.dirname(
                    rel_path), input_fname=fname + '.flac', transcript=
                    transcript))
    return input_data
