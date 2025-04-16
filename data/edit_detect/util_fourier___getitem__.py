def __getitem__(self, key: torch.Tensor) ->TensorDict:
    if self.__top_dict__.is_key_exist(key=key):
        return self.__top_dict__[key]
    else:
        self.__top_dict__[key] = TensorDict(max_size=self.__sub_dict_max_size__
            )
        return self.__top_dict__[key]
