def __del__(self):
    sys.stdout = self.__stdout
