def build_eval_dataset(filenames: List[Text], labels: List[int]=None,
    image_size: int=IMAGE_SIZE, batch_size: int=1) ->tf.Tensor:
    """Builds a tf.data.Dataset from a list of filenames and labels.

  Args:
    filenames: a list of filename paths of images.
    labels: a list of labels corresponding to each image.
    image_size: image height/width dimension.
    batch_size: the batch size used by the dataset

  Returns:
    A preprocessed and normalized image `Tensor`.
  """
    if labels is None:
        labels = [0] * len(filenames)
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(lambda filename, label: (load_eval_image(filename,
        image_size), label))
    dataset = dataset.batch(batch_size)
    return dataset
