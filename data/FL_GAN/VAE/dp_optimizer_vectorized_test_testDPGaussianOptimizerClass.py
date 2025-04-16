@parameterized.named_parameters(('DPGradientDescent', VectorizedDPSGD), (
    'DPAdagrad', VectorizedDPAdagrad), ('DPAdam', VectorizedDPAdam))
def testDPGaussianOptimizerClass(self, cls):
    with self.cached_session() as sess:
        var0 = tf.Variable([0.0])
        data0 = tf.Variable([[0.0]])
        opt = cls(l2_norm_clip=4.0, noise_multiplier=2.0, num_microbatches=
            1, learning_rate=2.0)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose([0.0], self.evaluate(var0))
        gradient_op = opt.compute_gradients(self._loss(data0, var0), [var0])
        grads = []
        for _ in range(1000):
            grads_and_vars = sess.run(gradient_op)
            grads.append(grads_and_vars[0][0])
        self.assertNear(np.std(grads), 2.0 * 4.0, 0.5)
