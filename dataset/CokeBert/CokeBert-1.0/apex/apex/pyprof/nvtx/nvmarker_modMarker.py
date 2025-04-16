def modMarker(mod, fn_name, args):
    """
	Returns the stringified extra_repr() of a module.
	"""
    assert fn_name == 'forward'
    assert len(args) > 0
    d = {}
    d['mod'] = mod.__name__
    d['strRepr'] = args[0].extra_repr()
    return str(d)
