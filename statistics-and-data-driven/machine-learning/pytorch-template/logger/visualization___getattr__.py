def __getattr__(self, name):
    """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
    if name in self.tb_writer_ftns:
        add_data = getattr(self.writer, name, None)

        def wrapper(tag, data, *args, **kwargs):
            if add_data is not None:
                if name not in self.tag_mode_exceptions:
                    tag = '{}/{}'.format(tag, self.mode)
                add_data(tag, data, self.step, *args, **kwargs)
        return wrapper
    else:
        try:
            attr = object.__getattr__(name)
        except AttributeError:
            raise AttributeError("type object '{}' has no attribute '{}'".
                format(self.selected_module, name))
        return attr
