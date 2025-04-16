def is_key_exist(self, key: torch.Tensor) ->bool:
    embed_key = self.__get_key__(key)
    return embed_key in self.__data__
