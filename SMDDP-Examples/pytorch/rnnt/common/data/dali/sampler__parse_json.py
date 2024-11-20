def _parse_json(json_path: str, start_label=0, predicate=lambda json: True,
    tokenized_transcript=False):
    """
    Parses json file to the format required by DALI
    Args:
        json_path: path to json file
        start_label: the label, starting from which DALI will assign consecutive int numbers to every transcript
        predicate: function, that accepts a sample descriptor (i.e. json dictionary) as an argument.
                   If the predicate for a given sample returns True, it will be included in the dataset.

    Returns:
        output_files: dictionary, that maps file name to label assigned by DALI
        transcripts: dictionary, that maps label assigned by DALI to the transcript
    """
    import json
    global cnt
    with open(json_path) as f:
        librispeech_json = json.load(f)
    output_files = {}
    transcripts = {}
    curr_label = start_label
    for original_sample in librispeech_json:
        if not predicate(original_sample):
            continue
        transcripts[curr_label] = original_sample['tokenized_transcript' if
            tokenized_transcript else 'transcript']
        output_files[original_sample['files'][-1]['fname']] = dict(label=
            curr_label, duration=original_sample['original_duration'])
        curr_label += 1
    return output_files, transcripts
