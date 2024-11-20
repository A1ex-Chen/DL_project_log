def _parse_pkl(pkl_path: str, start_label=0, predicate=lambda pkl: True,
    tokenized_transcript=True):
    if not tokenized_transcript:
        raise NotImplementedError(
            'pickle input only works with tokenized_transcript')
    import pickle
    with open(pkl_path, 'rb') as f:
        librispeech_pkl = pickle.load(f)
    output_files = {}
    transcripts = {}
    curr_label = start_label
    for original_sample in librispeech_pkl:
        if not predicate(original_sample):
            continue
        transcripts[curr_label] = original_sample['tokenized_transcript']
        output_files[original_sample['fname']] = dict(label=curr_label,
            duration=original_sample['original_duration'])
        curr_label += 1
    return output_files, transcripts
