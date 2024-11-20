def log(self, *args, **kwargs):
    if self.key in kwargs:
        self.writer.add_scalar(self.group_name + '/' + self.graph_label,
            kwargs[self.key], kwargs['accum_iter'])
    else:
        self.writer.add_scalar(self.group_name + '/' + self.graph_label, 0,
            kwargs['accum_iter'])
