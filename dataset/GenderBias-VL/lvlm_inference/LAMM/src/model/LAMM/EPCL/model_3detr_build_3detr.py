def build_3detr(args, dataset_config):
    print(args)
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    model = Model3DETR(pre_encoder, encoder, decoder, dataset_config, args.
        dataset_name, encoder_dim=args.enc_dim, decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout, num_queries=args.nqueries,
        use_task_emb=args.use_task_emb)
    output_processor = BoxProcessor(dataset_config)
    return model, output_processor
