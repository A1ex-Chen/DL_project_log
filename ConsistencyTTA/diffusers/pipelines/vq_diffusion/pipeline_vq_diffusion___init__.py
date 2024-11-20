def __init__(self, vqvae: VQModel, text_encoder: CLIPTextModel, tokenizer:
    CLIPTokenizer, transformer: Transformer2DModel, scheduler:
    VQDiffusionScheduler, learned_classifier_free_sampling_embeddings:
    LearnedClassifierFreeSamplingEmbeddings):
    super().__init__()
    self.register_modules(vqvae=vqvae, transformer=transformer,
        text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler,
        learned_classifier_free_sampling_embeddings=
        learned_classifier_free_sampling_embeddings)
