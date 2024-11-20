@property
def dummy_inputs(self) ->Dict[str, tf.Tensor]:
    """
        Dummy inputs to build the network.

        Returns:
            :obj:`Dict[str, tf.Tensor]`: The dummy inputs.
        """
    return {'input_ids': tf.constant(DUMMY_INPUTS)}
