def preprocess_image_files(directory_name, arch, batch_size, num_channels=3,
    dtype=tf.float32):
    image_size = get_image_size_from_model(arch)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(data_format=
        'channels_last')
    images = datagen.flow_from_directory(directory_name, class_mode=None,
        batch_size=batch_size, target_size=(image_size, image_size),
        shuffle=False)
    return images
