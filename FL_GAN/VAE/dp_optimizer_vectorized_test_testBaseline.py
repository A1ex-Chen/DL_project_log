@parameterized.named_parameters(('DPGradientDescent 1', VectorizedDPSGD, 1,
    [-2.5, -2.5]), ('DPGradientDescent 2', VectorizedDPSGD, 2, [-2.5, -2.5]
    ), ('DPGradientDescent 4', VectorizedDPSGD, 4, [-2.5, -2.5]), (
    'DPAdagrad 1', VectorizedDPAdagrad, 1, [-2.5, -2.5]), ('DPAdagrad 2',
    VectorizedDPAdagrad, 2, [-2.5, -2.5]), ('DPAdagrad 4',
    VectorizedDPAdagrad, 4, [-2.5, -2.5]), ('DPAdam 1', VectorizedDPAdam, 1,
    [-2.5, -2.5]), ('DPAdam 2', VectorizedDPAdam, 2, [-2.5, -2.5]), (
    'DPAdam 4', VectorizedDPAdam, 4, [-2.5, -2.5]))
def testBaseline(self, cls, num_microbatches, expected_answer):
    with self.cached_session() as sess:
        var0 = tf.Variable([1.0, 2.0])
        data0 = tf.Variable([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [-1.0, 0.0]])
        opt = cls(l2_norm_clip=1000000000.0, noise_multiplier=0.0,
            num_microbatches=num_microbatches, learning_rate=2.0)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        gradient_op = opt.compute_gradients(self._loss(data0, var0), [var0])
        grads_and_vars = sess.run(gradient_op)
        self.assertAllCloseAccordingToType(expected_answer, grads_and_vars[
            0][0])
