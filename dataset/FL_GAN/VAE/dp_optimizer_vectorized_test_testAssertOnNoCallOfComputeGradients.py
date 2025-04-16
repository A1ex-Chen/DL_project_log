@parameterized.named_parameters(('DPGradientDescent', VectorizedDPSGD), (
    'DPAdagrad', VectorizedDPAdagrad), ('DPAdam', VectorizedDPAdam))
def testAssertOnNoCallOfComputeGradients(self, cls):
    opt = cls(l2_norm_clip=4.0, noise_multiplier=2.0, num_microbatches=1,
        learning_rate=2.0)
    with self.assertRaises(AssertionError):
        grads_and_vars = tf.Variable([0.0])
        opt.apply_gradients(grads_and_vars)
    var0 = tf.Variable([0.0])
    data0 = tf.Variable([[0.0]])
    grads_and_vars = opt.compute_gradients(self._loss(data0, var0), [var0])
    opt.apply_gradients(grads_and_vars)
