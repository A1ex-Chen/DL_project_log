def sample(self, file_names, in_mem_file_list, tokenized_transcript):
    output_files, self.transcripts = {}, {}
    max_duration = self.config_data['max_duration']
    for file in file_names:
        if file.endswith('.json'):
            parse_func = _parse_json
        elif file.endswith('.pkl'):
            parse_func = _parse_pkl
        else:
            raise NotImplementedError(
                'Please supply supported input data file type: json or pickle')
        of, tr = parse_func(file if file[0] == '/' else os.path.join(
            dataset_path, file), len(output_files), predicate=lambda file: 
            file['original_duration'] <= max_duration, tokenized_transcript
            =tokenized_transcript)
        output_files.update(of)
        self.transcripts.update(tr)
    if in_mem_file_list:
        self.make_files(output_files)
    else:
        self.make_file_list(output_files, file_names)
