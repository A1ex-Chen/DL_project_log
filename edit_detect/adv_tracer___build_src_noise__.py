def __build_src_noise__(self):
    self.__build_noise__()
    self.__src_idx__ = self.__repeat_data__(data=[i for i in range(self.
        __unique_src_n__)])
    self.__src__ = self.__repeat_data__(data=self.__src__)
