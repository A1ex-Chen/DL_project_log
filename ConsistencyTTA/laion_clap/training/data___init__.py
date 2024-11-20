def __init__(self, index_path, ipc, config, eval_mode=False):
    """Toy Dataset for testing the audioset input with text labels
        Parameters
        ----------
            index_path: str
                the link to the h5 file of each audio
            idc: str
                the link to the npy file, the number of samples in each class
            config: dict
                the audio cfg file
           eval_model (bool): to indicate if the dataset is a testing dataset
        """
    self.audio_cfg = config['audio_cfg']
    self.text_cfg = config['text_cfg']
    self.fp = h5py.File(index_path, 'r')
    self.ipc = np.load(ipc, allow_pickle=True)
    self.total_size = len(self.fp['audio_name'])
    self.classes_num = self.audio_cfg['class_num']
    self.eval_mode = eval_mode
    if not eval_mode:
        self.generate_queue()
    else:
        self.queue = []
        for i in range(self.total_size):
            target = self.fp['target'][i]
            if np.sum(target) > 0:
                self.queue.append(i)
        self.total_size = len(self.queue)
    logging.info('total dataset size: %d' % self.total_size)
    logging.info('class num: %d' % self.classes_num)
