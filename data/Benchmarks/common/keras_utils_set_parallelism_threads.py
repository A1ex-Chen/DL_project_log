def set_parallelism_threads():
    """Set the number of parallel threads according to the number available on the hardware"""
    if (K.backend() == 'tensorflow' and 'NUM_INTRA_THREADS' in os.environ and
        'NUM_INTER_THREADS' in os.environ):
        import tensorflow as tf
        session_conf = tf.ConfigProto(inter_op_parallelism_threads=int(os.
            environ['NUM_INTER_THREADS']), intra_op_parallelism_threads=int
            (os.environ['NUM_INTRA_THREADS']))
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
