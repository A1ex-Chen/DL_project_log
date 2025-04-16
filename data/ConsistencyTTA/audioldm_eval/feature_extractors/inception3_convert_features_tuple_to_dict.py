def convert_features_tuple_to_dict(self, features):
    """
        The only compound return type of the forward function amenable to JIT tracing is tuple.
        This function simply helps to recover the mapping.
        """
    message = 'Features must be the output of forward function'
    assert type(features) is tuple and len(features) == len(self.features_list
        ), message
    return dict((name, feature) for name, feature in zip(self.features_list,
        features))
