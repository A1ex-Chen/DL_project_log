def _group_std(self, x):
    input_shape = tf.shape(x)
    N = input_shape[0]
    H = input_shape[1]
    W = input_shape[2]
    C = input_shape[3]
    num_groups = C // self.groups
    x = tf.reshape(x, [N, H, W, self.groups, num_groups])
    _, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
    std = tf.sqrt(var + self.eps)
    std = tf.broadcast_to(std, [N, H, W, self.groups, num_groups])
    return tf.reshape(std, input_shape)
