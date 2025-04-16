@classmethod
def from_pretrained(cls, retriever_name_or_path, indexed_dataset=None, **kwargs
    ):
    requires_datasets(cls)
    requires_faiss(cls)
    config = kwargs.pop('config', None) or RagConfig.from_pretrained(
        retriever_name_or_path, **kwargs)
    rag_tokenizer = RagTokenizer.from_pretrained(retriever_name_or_path,
        config=config)
    question_encoder_tokenizer = rag_tokenizer.question_encoder
    generator_tokenizer = rag_tokenizer.generator
    if indexed_dataset is not None:
        config.index_name = 'custom'
        index = CustomHFIndex(config.retrieval_vector_size, indexed_dataset)
    else:
        index = cls._build_index(config)
    return cls(config, question_encoder_tokenizer=
        question_encoder_tokenizer, generator_tokenizer=generator_tokenizer,
        index=index)
