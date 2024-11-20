def predict(self, to_predict, checkpoint_name=None, print_results=True):
    images = preprocess_image_files(directory_name=to_predict, arch=self.
        params.arch, batch_size=self.params.predict_batch_size, dtype=
        DTYPE_MAP[self.params.dtype])
    nb_samples = len(images)
    if checkpoint_name is not None:
        self.model.load_weights(checkpoint_name)
    try:
        file_names = images.filenames
        num_files = len(file_names)
        if self.params.benchmark:
            nb_samples *= 50
            print_results = False
            num_files *= 50
        start_time = time.time()
        inference_results = self.model.predict(images, verbose=1, steps=
            nb_samples)
        total_time = time.time() - start_time
        score = tf.nn.softmax(inference_results, axis=1)
        if print_results:
            for i, name in enumerate(file_names):
                print(
                    'This {} image most likely belongs to {} class with a {} percent confidence.'
                    .format(name, tf.math.argmax(score[i]), 100 * tf.math.
                    reduce_max(score[i])))
        print('Total time to infer {} images :: {}'.format(num_files,
            total_time))
        print('Inference Throughput {}'.format(num_files / total_time))
        print('Inference Latency {}'.format(total_time / num_files))
    except KeyboardInterrupt:
        print('Keyboard interrupt')
    print('Ending Inference ...')
