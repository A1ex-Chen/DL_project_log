def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
    y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)
    if max_val is not None and y > max_val:
        y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)
    if y < min_val:
        y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)
    return y
