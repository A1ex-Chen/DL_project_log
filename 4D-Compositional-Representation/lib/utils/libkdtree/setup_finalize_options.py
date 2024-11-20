def finalize_options(self):
    """
        In order to avoid premature import of numpy before it gets installed as a dependency
        get numpy include directories during the extensions building process
        http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
        """
    build_ext.finalize_options(self)
    set_builtin('__NUMPY_SETUP__', False)
    import numpy
    self.include_dirs.append(numpy.get_include())
