def unroll_gen_dataset_combination(self, combinations: Union[str, List[str],
    Dict[str, str]]):
    if combinations is None:
        return []
    elif isinstance(combinations, str):
        return [combinations]
    elif isinstance(combinations, list):
        unrolled_combs: List[str] = []
        for comb in combinations:
            unrolled_combs = (unrolled_combs + ModelDataset.
                unroll_dataset_combination(combinations=comb))
        return unrolled_combs
    elif isinstance(combinations, dict):
        unrolled_combs: List[str] = []
        all_models = self.filter_md_by(archs=None)
        for key, val in combinations.items():
            if key in all_models:
                unrolled_combs.append(ModelDataset.
                    __model_dataset_name_fn__(model=key, dataset=val))
            else:
                unrolled_combs.append(ModelDataset.
                    __model_dataset_name_fn__(model=val, dataset=key))
        return combinations
