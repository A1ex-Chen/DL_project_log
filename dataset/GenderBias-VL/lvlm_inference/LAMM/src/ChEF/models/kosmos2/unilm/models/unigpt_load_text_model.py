@classmethod
def load_text_model(cls, args, task):
    """Load a roberta model from the fairseq library."""
    if args.text_encoder == 'none':
        return None, None
    mlm_args = copy.deepcopy(args)
    mlm_task = task
    logger.info('Roberta dictionary: {} types'.format(len(mlm_task.dictionary))
        )
    mlm_args.layernorm_embedding = True
    mlm_args.no_scale_embedding = True
    mlm_args.dropout = 0.1
    mlm_args.attention_dropout = 0.1
    mlm_args.tokens_per_sample = mlm_args.mlm_tokens_per_sample
    mlm_model = RobertaModel.build_model(mlm_args, mlm_task)
    logger.info('mlm args is {}'.format(mlm_args))
    if args.mlm_model_path != '':
        state = checkpoint_utils.load_checkpoint_to_cpu(args.mlm_model_path)
        mlm_model.load_state_dict(state['model'], strict=True, args=mlm_args)
    connector = build_connector(args, args.encoder_embed_dim, args.
        decoder_embed_dim)
    return mlm_model, connector
