@run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
def encoder_decoder_forward():
    return model(input_ids, decoder_input_ids=input_ids, training=False)
