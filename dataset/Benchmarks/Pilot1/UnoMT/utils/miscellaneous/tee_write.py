def write(self, data):
    with open(self.__log_name, self.__mode) as file:
        file.write(data)
    self.__stdout.write(data)
