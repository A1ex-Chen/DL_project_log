def main(argv):
    if len(argv) >= 2:
        model = tf.keras.models.load_model(argv[1])
    else:
        model = mobilenet_v1()
    model.summary()
    batch_size = 50
    validation_split = 0.1
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range
        =10, width_shift_range=0.05, height_shift_range=0.05, zoom_range=
        0.1, horizontal_flip=True, validation_split=validation_split,
        rescale=1.0 / 255)
    train_generator = datagen.flow_from_directory(BASE_DIR, target_size=(
        IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, subset='training',
        color_mode='rgb')
    val_generator = datagen.flow_from_directory(BASE_DIR, target_size=(
        IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, subset='validation',
        color_mode='rgb')
    print(train_generator.class_indices)
    model = train_epochs(model, train_generator, val_generator, 20, 0.001)
    model = train_epochs(model, train_generator, val_generator, 10, 0.0005)
    model = train_epochs(model, train_generator, val_generator, 20, 0.00025)
    if len(argv) >= 3:
        model.save(argv[2])
    else:
        model.save('trained_models/vww_96.h5')
