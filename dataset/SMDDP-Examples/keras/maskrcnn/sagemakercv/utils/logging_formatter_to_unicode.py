@staticmethod
def to_unicode(value):
    """
        Converts a string argument to a unicode string.
        If the argument is already a unicode string or None, it is returned
        unchanged.  Otherwise it must be a byte string and is decoded as utf8.
        """
    try:
        if isinstance(value, (str, type(None))):
            return value
        if not isinstance(value, bytes):
            raise TypeError('Expected bytes, unicode, or None; got %r' %
                type(value))
        return value.decode('utf-8')
    except UnicodeDecodeError:
        return repr(value)
