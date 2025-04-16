def __getitem__(self, k):
    if isinstance(k, str):
        inner_dict = dict(self.items())
        return inner_dict[k]
    else:
        return self.to_tuple()[k]
