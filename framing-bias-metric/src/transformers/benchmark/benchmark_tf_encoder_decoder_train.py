@run_with_tf_optimizations(self.args.eager_mode, self.args.use_xla)
def encoder_decoder_train():
    loss = model(input_ids, decoder_input_ids=input_ids, labels=input_ids,
        training=True)[0]
    gradients = tf.gradients(loss, model.trainable_variables)
    return gradients
