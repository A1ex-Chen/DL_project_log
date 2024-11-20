def getString(self, id_):
    """
		Get the string associated with an id.
		"""
    cmd = 'select value from {} where _id_ = {}'.format(self.stringT, id_)
    result = self.db.select(cmd)
    assert len(result) == 1
    return result[0]['value']
