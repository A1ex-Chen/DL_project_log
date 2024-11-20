def load_single_step(self, files, points_dict, loc0, scale0):
    """ Loads data for a single step.

        Args:
            files (list): list of files
            points_dict (dict): points dictionary for first step of sequence
            loc0 (tuple): location of first time step mesh
            scale0 (float): scale of first time step mesh
        """
    if self.fixed_time_step is None:
        time_step = np.random.choice(self.seq_len)
    else:
        time_step = int(self.fixed_time_step)
    if time_step != 0:
        points_dict = np.load(files[time_step])
    points = points_dict['points'].astype(np.float32)
    occupancies = points_dict['occupancies']
    if self.unpackbits:
        occupancies = np.unpackbits(occupancies)[:points.shape[0]]
    occupancies = occupancies.astype(np.float32)
    if self.scale_type == 'oflow':
        loc = points_dict['loc'].astype(np.float32)
        scale = points_dict['scale'].astype(np.float32)
        points = (loc + scale * points - loc0) / scale0
    if self.seq_len > 1:
        time = np.array(time_step / (self.seq_len - 1), dtype=np.float32)
    else:
        time = np.array([1], dtype=np.float32)
    data = {None: points, 'occ': occupancies, 'time': time}
    return data
