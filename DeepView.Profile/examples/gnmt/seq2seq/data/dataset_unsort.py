def unsort(self, array):
    """
        "Unsorts" given array (restores original order of elements before
        dataset was sorted by sequence length).

        :param array: array to be "unsorted"
        """
    if self.sorted:
        inverse = sorted(enumerate(self.indices), key=itemgetter(1))
        array = [array[i[0]] for i in inverse]
    return array
