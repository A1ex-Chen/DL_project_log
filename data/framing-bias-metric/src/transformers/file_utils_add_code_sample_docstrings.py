def add_code_sample_docstrings(*docstr, tokenizer_class=None, checkpoint=
    None, output_type=None, config_class=None, mask=None):

    def docstring_decorator(fn):
        model_class = fn.__qualname__.split('.')[0]
        is_tf_class = model_class[:2] == 'TF'
        doc_kwargs = dict(model_class=model_class, tokenizer_class=
            tokenizer_class, checkpoint=checkpoint)
        if 'SequenceClassification' in model_class:
            code_sample = (TF_SEQUENCE_CLASSIFICATION_SAMPLE if is_tf_class
                 else PT_SEQUENCE_CLASSIFICATION_SAMPLE)
        elif 'QuestionAnswering' in model_class:
            code_sample = (TF_QUESTION_ANSWERING_SAMPLE if is_tf_class else
                PT_QUESTION_ANSWERING_SAMPLE)
        elif 'TokenClassification' in model_class:
            code_sample = (TF_TOKEN_CLASSIFICATION_SAMPLE if is_tf_class else
                PT_TOKEN_CLASSIFICATION_SAMPLE)
        elif 'MultipleChoice' in model_class:
            code_sample = (TF_MULTIPLE_CHOICE_SAMPLE if is_tf_class else
                PT_MULTIPLE_CHOICE_SAMPLE)
        elif 'MaskedLM' in model_class or model_class in [
            'FlaubertWithLMHeadModel', 'XLMWithLMHeadModel']:
            doc_kwargs['mask'] = '[MASK]' if mask is None else mask
            code_sample = (TF_MASKED_LM_SAMPLE if is_tf_class else
                PT_MASKED_LM_SAMPLE)
        elif 'LMHead' in model_class:
            code_sample = (TF_CAUSAL_LM_SAMPLE if is_tf_class else
                PT_CAUSAL_LM_SAMPLE)
        elif 'Model' in model_class or 'Encoder' in model_class:
            code_sample = (TF_BASE_MODEL_SAMPLE if is_tf_class else
                PT_BASE_MODEL_SAMPLE)
        else:
            raise ValueError(
                f"Docstring can't be built for model {model_class}")
        output_doc = _prepare_output_docstrings(output_type, config_class
            ) if output_type is not None else ''
        built_doc = code_sample.format(**doc_kwargs)
        fn.__doc__ = (fn.__doc__ or '') + ''.join(docstr
            ) + output_doc + built_doc
        return fn
    return docstring_decorator
