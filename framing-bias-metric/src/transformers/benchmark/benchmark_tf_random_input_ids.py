def random_input_ids(batch_size: int, sequence_length: int, vocab_size: int
    ) ->['tf.Tensor']:
    rng = random.Random()
    values = [rng.randint(0, vocab_size - 1) for i in range(batch_size *
        sequence_length)]
    return tf.constant(values, shape=(batch_size, sequence_length), dtype=
        tf.int32)
