def __init__(self, dropblock, start_value, stop_value, nr_steps):
    super(LinearScheduler, self).__init__()
    self.dropblock = dropblock
    self.i = 0
    self.drop_values = np.linspace(start=start_value, stop=stop_value, num=
        int(nr_steps))
