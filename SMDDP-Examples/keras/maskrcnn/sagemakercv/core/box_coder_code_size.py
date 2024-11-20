@abstractproperty
def code_size(self):
    """Return the size of each code.

    This number is a constant and should agree with the output of the `encode`
    op (e.g. if rel_codes is the output of self.encode(...), then it should have
    shape [N, code_size()]).  This abstractproperty should be overridden by
    implementations.

    Returns:
      an integer constant
    """
    pass
