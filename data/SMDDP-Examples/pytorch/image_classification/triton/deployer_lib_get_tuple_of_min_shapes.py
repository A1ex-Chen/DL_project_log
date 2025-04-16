def get_tuple_of_min_shapes(self, l):
    """ returns the tuple of min shapes 
            :: l :: list of tuples of tensors """
    shapes = self.get_list_of_shapes(l, min)
    min_batch = 1
    shapes = [[min_batch, *shape[1:]] for shape in shapes]
    shapes = tuple(shapes)
    return shapes
