def representative_dataset_gen():
    dataset_dir = os.path.join(BASE_DIR, 'person')
    for idx, image_file in enumerate(os.listdir(dataset_dir)):
        if idx > 10:
            return
        full_path = os.path.join(dataset_dir, image_file)
        if os.path.isfile(full_path):
            img = tf.keras.preprocessing.image.load_img(full_path,
                color_mode='rgb').resize((96, 96))
            arr = tf.keras.preprocessing.image.img_to_array(img)
            yield [arr.reshape(1, 96, 96, 3) / 255.0]
