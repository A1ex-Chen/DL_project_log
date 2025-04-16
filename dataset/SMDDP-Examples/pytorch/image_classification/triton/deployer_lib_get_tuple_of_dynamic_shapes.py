def get_tuple_of_dynamic_shapes(self, l):
    """ returns a tuple of dynamic shapes: variable tensor dimensions 
            (for ex. batch size) occur as -1 in the tuple
            :: l :: list of tuples of tensors """
    tensor_tuple = l[0]
    shapes = [list(x.shape) for x in tensor_tuple]
    for tensor_tuple in l:
        err_msg = 'tensors with varying shape lengths are not supported'
        assert len(tensor_tuple) == len(shapes), err_msg
        for i, x in enumerate(tensor_tuple):
            for j in range(len(x.shape)):
                if shapes[i][j] != x.shape[j] or j == 0:
                    shapes[i][j] = -1
    shapes = tuple(shapes)
    return shapes
