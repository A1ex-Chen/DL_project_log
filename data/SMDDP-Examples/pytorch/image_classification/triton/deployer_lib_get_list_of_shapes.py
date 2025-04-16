def get_list_of_shapes(self, l, fun):
    """ returns the list of min/max shapes, depending on fun
            :: l :: list of tuples of tensors
            :: fun :: min or max
        """
    tensor_tuple = l[0]
    shapes = [list(x.shape) for x in tensor_tuple]
    for tensor_tuple in l:
        assert len(tensor_tuple) == len(shapes
            ), 'tensors with varying shape lengths are not supported'
        for i, x in enumerate(tensor_tuple):
            for j in range(len(x.shape)):
                shapes[i][j] = fun(shapes[i][j], x.shape[j])
    return shapes
