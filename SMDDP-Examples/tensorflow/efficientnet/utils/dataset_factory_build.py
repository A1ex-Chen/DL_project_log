def build(self) ->tf.data.Dataset:
    """Construct a dataset end-to-end and return it.

    Args:
      input_context: An optional context provided by `tf.distribute` for
        cross-replica training.

    Returns:
      A TensorFlow dataset outputting batched images and labels.
    """
    if self._use_dali:
        print('Using dali for {train} dataloading'.format(train='training' if
            self.is_training else 'validation'))
        tfrec_filenames = sorted(tf.io.gfile.glob(os.path.join(self.
            _data_dir, '%s-*' % self._split)))
        tfrec_idx_filenames = sorted(tf.io.gfile.glob(os.path.join(self.
            _index_file, '%s-*' % self._split)))
        dali_pipeline = Dali.DaliPipeline(tfrec_filenames=tfrec_filenames,
            tfrec_idx_filenames=tfrec_idx_filenames, height=self.
            _image_size, width=self._image_size, batch_size=self.
            local_batch_size, num_threads=1, device_id=sdp.local_rank(),
            shard_id=sdp.rank(), num_gpus=sdp.size(), num_classes=self.
            num_classes, deterministic=False, dali_cpu=False, training=self
            .is_training)
        shapes = (self.local_batch_size, self._image_size, self._image_size, 3
            ), (self.local_batch_size, self._num_classes)
        dtypes = tf.float32, tf.float32
        dataset = dali_tf.DALIDataset(pipeline=dali_pipeline, batch_size=
            self.local_batch_size, output_shapes=shapes, output_dtypes=
            dtypes, device_id=sdp.local_rank())
        return dataset
    else:
        print('Using tf native pipeline for {train} dataloading'.format(
            train='training' if self.is_training else 'validation'))
        dataset = self.load_records()
        dataset = self.pipeline(dataset)
        return dataset
