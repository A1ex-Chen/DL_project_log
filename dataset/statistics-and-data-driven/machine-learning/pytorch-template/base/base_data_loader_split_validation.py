def split_validation(self):
    if self.valid_sampler is None:
        return None
    else:
        return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
