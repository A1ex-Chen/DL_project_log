def forward(self, t, y, **f_options):
    return self.base_func(t, y[0], **f_options),
