def prepare_inputs_for_generation(self, input_ids, past=None, use_cache=
    None, num_hashes=None, **kwargs):
    if past is not None:
        input_ids = input_ids[:, -1:]
    inputs_dict = {'input_ids': input_ids, 'past_buckets_states': past,
        'use_cache': use_cache, 'num_hashes': num_hashes}
    return inputs_dict
