def process_raw_data(self, fname, max_size):
    """
        Loads data from the input file.

        :param fname: input file name
        :param max_size: loads at most 'max_size' samples from the input file,
            if None loads the entire dataset
        """
    logging.info(f'Processing data from {fname}')
    data = []
    with open(fname) as dfile:
        for idx, line in enumerate(dfile):
            if max_size and idx == max_size:
                break
            data.append(line)
    return data
