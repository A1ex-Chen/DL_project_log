def __call__(self, values):
    return locate(self.class_name)(values[0])
