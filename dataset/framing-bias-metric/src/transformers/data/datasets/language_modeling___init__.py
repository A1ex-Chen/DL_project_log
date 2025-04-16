def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str,
    block_size: int, overwrite_cache=False, short_seq_probability=0.1,
    nsp_probability=0.5):
    warnings.warn(DEPRECATION_WARNING.format(
        'https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py'
        ), FutureWarning)
    assert os.path.isfile(file_path), f'Input file path {file_path} not found'
    self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair
        =True)
    self.short_seq_probability = short_seq_probability
    self.nsp_probability = nsp_probability
    directory, filename = os.path.split(file_path)
    cached_features_file = os.path.join(directory, 'cached_nsp_{}_{}_{}'.
        format(tokenizer.__class__.__name__, str(block_size), filename))
    self.tokenizer = tokenizer
    lock_path = cached_features_file + '.lock'
    with FileLock(lock_path):
        if os.path.exists(cached_features_file) and not overwrite_cache:
            start = time.time()
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
            logger.info(
                f'Loading features from cached file {cached_features_file} [took %.3f s]'
                , time.time() - start)
        else:
            logger.info(f'Creating features from dataset file at {directory}')
            self.documents = [[]]
            with open(file_path, encoding='utf-8') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line and len(self.documents[-1]) != 0:
                        self.documents.append([])
                    tokens = tokenizer.tokenize(line)
                    tokens = tokenizer.convert_tokens_to_ids(tokens)
                    if tokens:
                        self.documents[-1].append(tokens)
            logger.info(
                f'Creating examples from {len(self.documents)} documents.')
            self.examples = []
            for doc_index, document in enumerate(self.documents):
                self.create_examples_from_document(document, doc_index)
            start = time.time()
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.
                    HIGHEST_PROTOCOL)
            logger.info('Saving features into cached file %s [took %.3f s]',
                cached_features_file, time.time() - start)
