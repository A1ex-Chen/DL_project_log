def get_tuple_of_opt_shapes(self, l):
    """ returns the tuple of opt shapes 
            :: l :: list of tuples of tensors """
    counter = Counter()
    for tensor_tuple in l:
        shapes = [tuple(x.shape) for x in tensor_tuple]
        shapes = tuple(shapes)
        counter[shapes] += 1
    shapes = counter.most_common(1)[0][0]
    return shapes
