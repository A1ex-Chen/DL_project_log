def transform(model, inputs, outputs, output_format):
    for i, o in zip(inputs, outputs):
        with open(i, 'r') as f:
            j = json.load(f)
        if output_format == 'json':
            for entry in j:
                entry['tokenized_transcript'] = model.encode(entry[
                    'transcript'])
            with open(o, 'w') as f:
                json.dump(j, f)
        else:
            pruned_j = []
            for entry in j:
                pruned_entry = {'fname': entry['files'][-1]['fname'],
                    'original_duration': entry['original_duration'],
                    'tokenized_transcript': model.encode(entry['transcript'])}
                pruned_j.append(pruned_entry)
            with open(o, 'wb') as f:
                pickle.dump(pruned_j, f)
