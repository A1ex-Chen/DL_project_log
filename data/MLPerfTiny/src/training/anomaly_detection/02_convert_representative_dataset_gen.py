def representative_dataset_gen():
    for sample in train_data[::5]:
        sample = numpy.expand_dims(sample.astype(numpy.float32), axis=0)
        yield [sample]
