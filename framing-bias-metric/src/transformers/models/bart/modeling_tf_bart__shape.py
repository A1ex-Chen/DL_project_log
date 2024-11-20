def _shape(self, tensor: tf.Tensor, dim_0, bsz) ->tf.Tensor:
    reshaped_T_B_D = tf.reshape(tensor, (dim_0, bsz * self.num_heads, self.
        head_dim))
    return tf.transpose(reshaped_T_B_D, perm=(1, 0, 2))
