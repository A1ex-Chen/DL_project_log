def demangle(name):
    """
	Demangle a C++ string
	"""
    return cxxfilt.demangle(name)
