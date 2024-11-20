def is_shape_dynamic(self, shape):
    return any([self.is_dimension_dynamic(dim) for dim in shape])
