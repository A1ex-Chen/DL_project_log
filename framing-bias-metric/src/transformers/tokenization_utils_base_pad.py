def pad(self, encoded_inputs: Union[BatchEncoding, List[BatchEncoding],
    Dict[str, EncodedInput], Dict[str, List[EncodedInput]], List[Dict[str,
    EncodedInput]]], padding: Union[bool, str, PaddingStrategy]=True,
    max_length: Optional[int]=None, pad_to_multiple_of: Optional[int]=None,
    return_attention_mask: Optional[bool]=None, return_tensors: Optional[
    Union[str, TensorType]]=None, verbose: bool=True) ->BatchEncoding:
    """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with ``self.padding_side``,
        ``self.pad_token_id`` and ``self.pad_token_type_id``)

        .. note::

            If the ``encoded_inputs`` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
            result will use the same type unless you provide a different tensor type with ``return_tensors``. In the
            case of PyTorch tensors, you will lose the specific device of your tensors however.

        Args:
            encoded_inputs (:class:`~transformers.BatchEncoding`, list of :class:`~transformers.BatchEncoding`, :obj:`Dict[str, List[int]]`, :obj:`Dict[str, List[List[int]]` or :obj:`List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input (:class:`~transformers.BatchEncoding` or :obj:`Dict[str,
                List[int]]`) or a batch of tokenized inputs (list of :class:`~transformers.BatchEncoding`, `Dict[str,
                List[List[int]]]` or `List[Dict[str, List[int]]]`) so you can use this method during preprocessing as
                well as in a PyTorch Dataloader collate function.

                Instead of :obj:`List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors),
                see the note above for the return type.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are attention masks? <../glossary.html#attention-mask>`__
            return_tensors (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            verbose (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to print more information and warnings.
        """
    if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs
        [0], (dict, BatchEncoding)):
        encoded_inputs = {key: [example[key] for example in encoded_inputs] for
            key in encoded_inputs[0].keys()}
    assert 'input_ids' in encoded_inputs, 'You should supply an encoding or a list of encodings to this method. An encoding is the output of one the encoding methods of the tokenizer, i.e. __call__/encode_plus/batch_encode_plus. '
    if not encoded_inputs['input_ids']:
        if return_attention_mask:
            encoded_inputs['attention_mask'] = []
        return encoded_inputs
    first_element = encoded_inputs['input_ids'][0]
    if isinstance(first_element, (list, tuple)) and first_element:
        first_element = first_element[0]
    if not isinstance(first_element, int):
        if is_tf_available() and isinstance(first_element, tf.Tensor):
            return_tensors = 'tf' if return_tensors is None else return_tensors
        elif is_torch_available() and isinstance(first_element, torch.Tensor):
            return_tensors = 'pt' if return_tensors is None else return_tensors
        elif isinstance(first_element, np.ndarray):
            return_tensors = 'np' if return_tensors is None else return_tensors
        else:
            raise ValueError(
                f'type of {first_element} unknown: {type(first_element)}. Should be one of a python, numpy, pytorch or tensorflow object.'
                )
        for key, value in encoded_inputs.items():
            encoded_inputs[key] = to_py_obj(value)
    padding_strategy, _, max_length, _ = (self.
        _get_padding_truncation_strategies(padding=padding, max_length=
        max_length, verbose=verbose))
    if encoded_inputs['input_ids'] and not isinstance(encoded_inputs[
        'input_ids'][0], (list, tuple)):
        encoded_inputs = self._pad(encoded_inputs, max_length=max_length,
            padding_strategy=padding_strategy, pad_to_multiple_of=
            pad_to_multiple_of, return_attention_mask=return_attention_mask)
        return BatchEncoding(encoded_inputs, tensor_type=return_tensors)
    batch_size = len(encoded_inputs['input_ids'])
    assert all(len(v) == batch_size for v in encoded_inputs.values()
        ), 'Some items in the output dictionary have a different batch size than others.'
    if padding_strategy == PaddingStrategy.LONGEST:
        max_length = max(len(inputs) for inputs in encoded_inputs['input_ids'])
        padding_strategy = PaddingStrategy.MAX_LENGTH
    batch_outputs = {}
    for i in range(batch_size):
        inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
        outputs = self._pad(inputs, max_length=max_length, padding_strategy
            =padding_strategy, pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask)
        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)
    return BatchEncoding(batch_outputs, tensor_type=return_tensors)
