@staticmethod
def read_calib(calib_path):
    """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
    calib_all = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if line == '\n':
                break
            key, value = line.split(':', 1)
            calib_all[key] = np.array([float(x) for x in value.split()])
    calib_out = {}
    calib_out['P2'] = calib_all['P2'].reshape(3, 4)
    calib_out['Tr'] = np.identity(4)
    calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)
    return calib_out
