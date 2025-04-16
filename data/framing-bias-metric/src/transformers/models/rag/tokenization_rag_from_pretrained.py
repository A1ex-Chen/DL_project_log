@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    from ..auto.tokenization_auto import AutoTokenizer
    config = kwargs.pop('config', None)
    if config is None:
        config = RagConfig.from_pretrained(pretrained_model_name_or_path)
    question_encoder = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, config=config.question_encoder,
        subfolder='question_encoder_tokenizer')
    generator = AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
        config=config.generator, subfolder='generator_tokenizer')
    return cls(question_encoder=question_encoder, generator=generator)
