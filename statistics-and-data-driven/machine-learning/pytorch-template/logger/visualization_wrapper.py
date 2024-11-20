def wrapper(tag, data, *args, **kwargs):
    if add_data is not None:
        if name not in self.tag_mode_exceptions:
            tag = '{}/{}'.format(tag, self.mode)
        add_data(tag, data, self.step, *args, **kwargs)
