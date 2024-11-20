def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
    copy_attr(self.ema, model, include, exclude)
