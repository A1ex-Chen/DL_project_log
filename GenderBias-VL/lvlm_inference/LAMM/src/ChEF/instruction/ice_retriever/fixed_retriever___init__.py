def __init__(self, train_dataset, test_dataset, ice_num: Optional[int]=1,
    seed: Optional[int]=43, ice_assigned_ids=None, **kwargs) ->None:
    super().__init__(train_dataset, test_dataset, seed, ice_num)
    self.ice_assigned_ids = ice_assigned_ids
