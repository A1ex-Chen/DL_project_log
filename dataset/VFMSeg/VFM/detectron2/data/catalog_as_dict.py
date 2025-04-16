def as_dict(self):
    """
        Returns all the metadata as a dict.
        Note that modifications to the returned dict will not reflect on the Metadata object.
        """
    return copy.copy(self.__dict__)
