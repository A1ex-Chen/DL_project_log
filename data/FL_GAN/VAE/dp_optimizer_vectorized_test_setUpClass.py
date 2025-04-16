@classmethod
def setUpClass(cls):
    super(DPOptimizerTest, cls).setUpClass()
    tf.compat.v1.disable_eager_execution()
