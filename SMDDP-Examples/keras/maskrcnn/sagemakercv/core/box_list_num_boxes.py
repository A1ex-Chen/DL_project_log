def num_boxes(self):
    """Returns number of boxes held in collection.

    Returns:
      a tensor representing the number of boxes held in the collection.
    """
    return tf.shape(input=self.data['boxes'])[0]
