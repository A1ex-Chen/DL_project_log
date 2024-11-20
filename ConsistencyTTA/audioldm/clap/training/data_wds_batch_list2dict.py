def wds_batch_list2dict(batch, keys=['__url__', '__key__', 'waveform',
    'text', 'raw_text', 'audio_name', 'text_name', 'audio_orig_sr']):
    """
    Return a dictionary of the batch, with keys as the names of the fields.
    """
    assert len(keys) == len(batch
        ), 'batch must have same number of keys as keys argument'
    return {keys[i]: batch[i] for i in range(len(batch))}
