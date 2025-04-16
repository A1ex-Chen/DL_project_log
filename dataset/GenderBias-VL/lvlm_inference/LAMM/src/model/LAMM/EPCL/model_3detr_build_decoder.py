def build_decoder(args):
    decoder_layer = TransformerDecoderLayer(d_model=args.dec_dim, nhead=
        args.dec_nhead, dim_feedforward=args.dec_ffn_dim, dropout=args.
        dec_dropout)
    decoder = TransformerDecoder(decoder_layer, num_layers=args.dec_nlayers,
        return_intermediate=True)
    return decoder
