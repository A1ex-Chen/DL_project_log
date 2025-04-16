def __init__(self, log_name, mode='a'):
    self.__stdout = sys.stdout
    self.__log_name = log_name
    self.__mode = mode
    try:
        os.makedirs(os.path.dirname(log_name))
    except FileExistsError:
        pass
