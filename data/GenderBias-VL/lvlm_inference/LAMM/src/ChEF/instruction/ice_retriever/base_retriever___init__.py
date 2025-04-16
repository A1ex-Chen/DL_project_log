def __init__(self, train_dataset, test_dataset, seed: Optional[int]=43,
    ice_num: Optional[int]=1, **kwargs) ->None:
    self.ice_num = ice_num
    self.index_ds = train_dataset
    self.test_ds = test_dataset
    self.seed = seed
    self.fixed_ice = None
