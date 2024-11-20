def get_tuple_of_max_shapes(self, l):
    """ returns the tuple of max shapes 
            :: l :: list of tuples of tensors """
    shapes = self.get_list_of_shapes(l, max)
    max_batch = max(2, shapes[0][0])
    shapes = [[max_batch, *shape[1:]] for shape in shapes]
    shapes = tuple(shapes)
    return shapes
