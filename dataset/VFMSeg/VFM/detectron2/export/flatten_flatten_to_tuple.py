def flatten_to_tuple(obj):
    """
    Flatten an object so it can be used for PyTorch tracing.
    Also returns how to rebuild the original object from the flattened outputs.

    Returns:
        res (tuple): the flattened results that can be used as tracing outputs
        schema: an object with a ``__call__`` method such that ``schema(res) == obj``.
             It is a pure dataclass that can be serialized.
    """
    schemas = [((str, bytes), IdentitySchema), (list, ListSchema), (tuple,
        TupleSchema), (collections.abc.Mapping, DictSchema), (Instances,
        InstancesSchema), ((Boxes, ROIMasks), TensorWrapSchema)]
    for klass, schema in schemas:
        if isinstance(obj, klass):
            F = schema
            break
    else:
        F = IdentitySchema
    return F.flatten(obj)
