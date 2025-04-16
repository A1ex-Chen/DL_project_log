def loadQA(self, ids=[]):
    """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
    if type(ids) == list:
        return [self.qa[id] for id in ids]
    elif type(ids) == int:
        return [self.qa[ids]]
