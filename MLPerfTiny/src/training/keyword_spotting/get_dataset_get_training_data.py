def get_training_data(Flags, get_waves=False, val_cal_subset=False):
    label_count = 12
    background_frequency = Flags.background_frequency
    background_volume_range_ = Flags.background_volume
    model_settings = models.prepare_model_settings(label_count, Flags)
    bg_path = Flags.bg_path
    BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
    background_data = prepare_background_data(bg_path,
        BACKGROUND_NOISE_DIR_NAME)
    splits = ['train', 'test', 'validation']
    (ds_train, ds_test, ds_val), ds_info = tfds.load('speech_commands',
        split=splits, data_dir=Flags.data_dir, with_info=True)
    if val_cal_subset:
        with open('quant_cal_idxs.txt') as fpi:
            cal_indices = [int(line) for line in fpi]
        cal_indices.sort()
        count = 0
        val_sub_audio = []
        val_sub_labels = []
        for d in ds_val:
            if count in cal_indices:
                new_audio = d['audio'].numpy()
                if len(new_audio) < 16000:
                    new_audio = np.pad(new_audio, (0, 16000 - len(new_audio
                        )), 'constant')
                val_sub_audio.append(new_audio)
                val_sub_labels.append(d['label'].numpy())
            count += 1
        ds_val = tf.data.Dataset.from_tensor_slices({'audio': val_sub_audio,
            'label': val_sub_labels})
    if Flags.num_train_samples != -1:
        ds_train = ds_train.take(Flags.num_train_samples)
    if Flags.num_val_samples != -1:
        ds_val = ds_val.take(Flags.num_val_samples)
    if Flags.num_test_samples != -1:
        ds_test = ds_test.take(Flags.num_test_samples)
    if get_waves:
        ds_train = ds_train.map(cast_and_pad)
        ds_test = ds_test.map(cast_and_pad)
        ds_val = ds_val.map(cast_and_pad)
    else:
        ds_train = ds_train.map(get_preprocess_audio_func(model_settings,
            is_training=True, background_data=background_data),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(get_preprocess_audio_func(model_settings,
            is_training=False, background_data=background_data),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(get_preprocess_audio_func(model_settings,
            is_training=False, background_data=background_data),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.map(convert_dataset)
        ds_test = ds_test.map(convert_dataset)
        ds_val = ds_val.map(convert_dataset)
    ds_train = ds_train.batch(Flags.batch_size)
    ds_test = ds_test.batch(Flags.batch_size)
    ds_val = ds_val.batch(Flags.batch_size)
    return ds_train, ds_test, ds_val
