def __getstate__(self) ->Dict:
    state = self.__dict__.copy()
    state.update({k: None for k in ['spm_source', 'spm_target',
        'current_spm', 'punc_normalizer']})
    return state
