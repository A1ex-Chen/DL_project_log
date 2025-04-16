@property
def dummy_inputs(self):
    """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
    return {'input_ids': tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS)}
