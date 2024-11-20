@parameterized.named_parameters(('DPGradientDescent', VectorizedDPSGD), (
    'DPAdagrad', VectorizedDPAdagrad), ('DPAdam', VectorizedDPAdam))
def testClippingNorm(self, cls):
    with self.cached_session() as sess:
        var0 = tf.Variable([0.0, 0.0])
        data0 = tf.Variable([[3.0, 4.0], [6.0, 8.0]])
        opt = cls(l2_norm_clip=1.0, noise_multiplier=0.0, num_microbatches=
            1, learning_rate=2.0)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose([0.0, 0.0], self.evaluate(var0))
        gradient_op = opt.compute_gradients(self._loss(data0, var0), [var0])
        grads_and_vars = sess.run(gradient_op)
        self.assertAllCloseAccordingToType([-0.6, -0.8], grads_and_vars[0][0])
