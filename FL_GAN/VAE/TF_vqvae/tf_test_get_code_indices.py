def get_code_indices(self, flattened_inputs):
    similarity = tf.matmul(flattened_inputs, self.embeddings)
    distances = tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True
        ) + tf.reduce_sum(self.embeddings ** 2, axis=0) - 2 * similarity
    encoding_indices = tf.argmin(distances, axis=1)
    return encoding_indices
