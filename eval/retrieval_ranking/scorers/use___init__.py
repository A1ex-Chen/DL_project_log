def __init__(self, scorer_type='use-v5', model_fpath='', use_cache=False):
    """ """
    self.logger = CreateLogger()
    self.logger.debug('[model]: USEScorer')
    self.logger.debug('[scorer_type]: %s', scorer_type)
    self.logger.debug('[model_fpath]: %s', model_fpath)
    self.logger.debug('[batch_size]: %d', MAX_BATCH_USE)
    self.logger.debug('[use_cache]: %s', use_cache)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    self.logger.info('USE type: %s', scorer_type)
    target_path = os.path.join(ROOT_DIR, '../../models/use-v5/')
    os.environ['TFHUB_CACHE_DIR'] = target_path
    self.logger.info('Cache_dir = %s', target_path)
    if os.path.isdir(target_path):
        self.logger.info('[cached] Skip downloading the model')
    else:
        self.logger.info('Download model (USE-v5)')
        if os.path.isdir(target_path) == False:
            os.makedirs(target_path)
    self.model_name = 'universal-sentence-encoder-large-v5'
    self.model = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5')
