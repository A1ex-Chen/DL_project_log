def images_to_tensorboard(self, runner):
    detection_images = self.build_image_batch(runner)
    writer = tf.summary.create_file_writer(runner.tensorboard_dir)
    with writer.as_default():
        for i, image in enumerate(detection_images):
            tf.summary.image(f'image_{i}', image, step=runner.iter)
    writer.close()
