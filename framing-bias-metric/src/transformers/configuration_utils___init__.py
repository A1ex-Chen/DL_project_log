def __init__(self, **kwargs):
    self.return_dict = kwargs.pop('return_dict', True)
    self.output_hidden_states = kwargs.pop('output_hidden_states', False)
    self.output_attentions = kwargs.pop('output_attentions', False)
    self.torchscript = kwargs.pop('torchscript', False)
    self.use_bfloat16 = kwargs.pop('use_bfloat16', False)
    self.pruned_heads = kwargs.pop('pruned_heads', {})
    self.tie_word_embeddings = kwargs.pop('tie_word_embeddings', True)
    self.is_encoder_decoder = kwargs.pop('is_encoder_decoder', False)
    self.is_decoder = kwargs.pop('is_decoder', False)
    self.add_cross_attention = kwargs.pop('add_cross_attention', False)
    self.tie_encoder_decoder = kwargs.pop('tie_encoder_decoder', False)
    self.max_length = kwargs.pop('max_length', 20)
    self.min_length = kwargs.pop('min_length', 0)
    self.do_sample = kwargs.pop('do_sample', False)
    self.early_stopping = kwargs.pop('early_stopping', False)
    self.num_beams = kwargs.pop('num_beams', 1)
    self.temperature = kwargs.pop('temperature', 1.0)
    self.top_k = kwargs.pop('top_k', 50)
    self.top_p = kwargs.pop('top_p', 1.0)
    self.repetition_penalty = kwargs.pop('repetition_penalty', 1.0)
    self.length_penalty = kwargs.pop('length_penalty', 1.0)
    self.no_repeat_ngram_size = kwargs.pop('no_repeat_ngram_size', 0)
    self.bad_words_ids = kwargs.pop('bad_words_ids', None)
    self.num_return_sequences = kwargs.pop('num_return_sequences', 1)
    self.chunk_size_feed_forward = kwargs.pop('chunk_size_feed_forward', 0)
    self.architectures = kwargs.pop('architectures', None)
    self.finetuning_task = kwargs.pop('finetuning_task', None)
    self.id2label = kwargs.pop('id2label', None)
    self.label2id = kwargs.pop('label2id', None)
    if self.id2label is not None:
        kwargs.pop('num_labels', None)
        self.id2label = dict((int(key), value) for key, value in self.
            id2label.items())
    else:
        self.num_labels = kwargs.pop('num_labels', 2)
    self.tokenizer_class = kwargs.pop('tokenizer_class', None)
    self.prefix = kwargs.pop('prefix', None)
    self.bos_token_id = kwargs.pop('bos_token_id', None)
    self.pad_token_id = kwargs.pop('pad_token_id', None)
    self.eos_token_id = kwargs.pop('eos_token_id', None)
    self.sep_token_id = kwargs.pop('sep_token_id', None)
    self.decoder_start_token_id = kwargs.pop('decoder_start_token_id', None)
    self.task_specific_params = kwargs.pop('task_specific_params', None)
    self.xla_device = kwargs.pop('xla_device', None)
    self._name_or_path = str(kwargs.pop('name_or_path', ''))
    for key, value in kwargs.items():
        try:
            setattr(self, key, value)
        except AttributeError as err:
            logger.error("Can't set {} with value {} for {}".format(key,
                value, self))
            raise err