def genetate_ice(self, ice_indices: List[List[int]], prompt: List[str],
    inferencer_type: str='default'):
    ice_batch = []
    for indices in ice_indices:
        ices = []
        for i in indices:
            ice = self.index_ds[i]
            if 'question' not in ice:
                if inferencer_type == 'icl_ppl':
                    ice['question'] = prompt[0][0]
                else:
                    ice['question'] = prompt[0]
            ices.append(ice)
        ice_batch.append(ices)
    return ice_batch
