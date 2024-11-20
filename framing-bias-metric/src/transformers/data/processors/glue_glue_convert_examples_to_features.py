def glue_convert_examples_to_features(examples: Union[List[InputExample],
    'tf.data.Dataset'], tokenizer: PreTrainedTokenizer, max_length:
    Optional[int]=None, task=None, label_list=None, output_mode=None):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset`` containing the
        task-specific features. If the input is a list of ``InputExamples``, will return a list of task-specific
        ``InputFeatures`` which can be fed to the model.

    """
    warnings.warn(DEPRECATION_WARNING.format('function'), FutureWarning)
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        if task is None:
            raise ValueError(
                'When calling glue_convert_examples_to_features from TF, the task parameter is required.'
                )
        return _tf_glue_convert_examples_to_features(examples, tokenizer,
            max_length=max_length, task=task)
    return _glue_convert_examples_to_features(examples, tokenizer,
        max_length=max_length, task=task, label_list=label_list,
        output_mode=output_mode)
