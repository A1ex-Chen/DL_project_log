@classmethod
def build_model(cls, args, task):
    model = TransformerLanguageModel.build_model(args, task)
    if getattr(args, 'max_target_positions', None) is None:
        args.max_target_positions = getattr(args, 'tokens_per_sample',
            DEFAULT_MAX_TARGET_POSITIONS)
    embed_tokens = cls.build_embedding(args, task.source_dictionary, args.
        decoder_embed_dim)
    embed_positions = PositionalEmbedding(args.max_target_positions, args.
        decoder_embed_dim, task.dictionary.pad(), learned=args.
        decoder_learned_pos
        ) if not args.no_token_positional_embeddings else None
    if args.share_decoder_input_output_embed:
        output_projection = torch.nn.Linear(embed_tokens.weight.shape[1],
            embed_tokens.weight.shape[0], bias=False)
        output_projection.weight = embed_tokens.weight
    else:
        output_projection = torch.nn.Linear(args.decoder_embed_dim, len(
            task.dictionary), bias=False)
        torch.nn.init.normal_(output_projection.weight, mean=0, std=args.
            decoder_embed_dim ** -0.5)
    if getattr(args, 'moe_freq', 0) > 0 and (getattr(args, 'fp16', False) and
        not getattr(args, 'memory_efficient_fp16', False) and getattr(args,
        'ddp_backend', None) != 'fully_sharded'):
        assert args.fp16_no_flatten_grads, 'If training moe models, set --fp16-no-flatten-grads to calculate correct gradnorm'
    args.ddp_rank = distributed_utils.get_data_parallel_rank()
    config = DecoderConfig()
    config.override(args)
    decoder = LMDecoder(config, embed_tokens, embed_positions,
        output_projection, is_encoder_decoder=False, dictionary=task.dictionary
        )
    decoder.chunk_emb = None
    if args.max_chunk_emb > 0:
        decoder.chunk_emb = TextEmbedding(args.max_chunk_emb, args.
            decoder_embed_dim)
    decoder.segment_emb = None
    if args.segment_emb:
        decoder.segment_emb = TextEmbedding(2, args.decoder_embed_dim)
    model.decoder = decoder
    if args.gpt_model_path != '':
        assert NotImplementedError
    return model
