def __init__(self, train_dataset, test_dataset, ice_num: Optional[int]=1,
    seed: Optional[int]=43, **kwargs) ->None:
    super().__init__(train_dataset, test_dataset, seed, ice_num)
