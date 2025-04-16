def getLayerName(mlist):
    """
			Get layer names from layer marker list.
			"""
    layers = []
    assert type(mlist) == list
    for m in mlist:
        assert 'layer:' in m
        l = m.split(':')[1]
        layers.append(l)
    return layers
