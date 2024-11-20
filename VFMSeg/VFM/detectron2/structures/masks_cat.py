@staticmethod
def cat(polymasks_list: List['PolygonMasks']) ->'PolygonMasks':
    """
        Concatenates a list of PolygonMasks into a single PolygonMasks

        Arguments:
            polymasks_list (list[PolygonMasks])

        Returns:
            PolygonMasks: the concatenated PolygonMasks
        """
    assert isinstance(polymasks_list, (list, tuple))
    assert len(polymasks_list) > 0
    assert all(isinstance(polymask, PolygonMasks) for polymask in
        polymasks_list)
    cat_polymasks = type(polymasks_list[0])(list(itertools.chain.
        from_iterable(pm.polygons for pm in polymasks_list)))
    return cat_polymasks
