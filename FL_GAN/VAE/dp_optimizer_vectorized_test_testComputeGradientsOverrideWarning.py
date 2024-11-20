@unittest.mock.patch('absl.logging.warning')
def testComputeGradientsOverrideWarning(self, mock_logging):


    class SimpleOptimizer(tf.compat.v1.train.Optimizer):

        def compute_gradients(self):
            return 0
    dp_optimizer_vectorized.make_vectorized_optimizer_class(SimpleOptimizer)
    mock_logging.assert_called_once_with(
        'WARNING: Calling make_optimizer_class() on class %s that overrides method compute_gradients(). Check to ensure that make_optimizer_class() does not interfere with overridden version.'
        , 'SimpleOptimizer')
