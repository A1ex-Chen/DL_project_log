def get_preprocess_audio_func(model_settings, is_training=False,
    background_data=[]):

    def prepare_processing_graph(next_element):
        """Builds a TensorFlow graph to apply the input distortions.
    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.
    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:
      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_padding_placeholder_: Where to pad the clip.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - mfcc_: Output 2D fingerprint of processed audio.
    Args:
      model_settings: Information about the current model being trained.
    """
        desired_samples = model_settings['desired_samples']
        background_frequency = model_settings['background_frequency']
        background_volume_range_ = model_settings['background_volume_range_']
        wav_decoder = tf.cast(next_element['audio'], tf.float32)
        if model_settings['feature_type'] != 'td_samples':
            wav_decoder = wav_decoder / tf.reduce_max(wav_decoder)
        else:
            wav_decoder = wav_decoder / tf.constant(2 ** 15, dtype=tf.float32)
        wav_decoder = tf.pad(wav_decoder, [[0, desired_samples - tf.shape(
            wav_decoder)[-1]]])
        foreground_volume_placeholder_ = tf.constant(1, dtype=tf.float32)
        scaled_foreground = tf.multiply(wav_decoder,
            foreground_volume_placeholder_)
        time_shift_padding_placeholder_ = tf.constant([[2, 2]], tf.int32)
        time_shift_offset_placeholder_ = tf.constant([2], tf.int32)
        scaled_foreground.shape
        padded_foreground = tf.pad(scaled_foreground,
            time_shift_padding_placeholder_, mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground,
            time_shift_offset_placeholder_, [desired_samples])
        if is_training and background_data != []:
            background_volume_range = tf.constant(background_volume_range_,
                dtype=tf.float32)
            background_index = np.random.randint(len(background_data))
            background_samples = background_data[background_index]
            background_offset = np.random.randint(0, len(background_samples
                ) - desired_samples)
            background_clipped = background_samples[background_offset:
                background_offset + desired_samples]
            background_clipped = tf.squeeze(background_clipped)
            background_reshaped = tf.pad(background_clipped, [[0, 
                desired_samples - tf.shape(wav_decoder)[-1]]])
            background_reshaped = tf.cast(background_reshaped, tf.float32)
            if np.random.uniform(0, 1) < background_frequency:
                background_volume = np.random.uniform(0,
                    background_volume_range_)
            else:
                background_volume = 0
            background_volume_placeholder_ = tf.constant(background_volume,
                dtype=tf.float32)
            background_data_placeholder_ = background_reshaped
            background_mul = tf.multiply(background_data_placeholder_,
                background_volume_placeholder_)
            background_add = tf.add(background_mul, sliced_foreground)
            sliced_foreground = tf.clip_by_value(background_add, -1.0, 1.0)
        if model_settings['feature_type'] == 'mfcc':
            stfts = tf.signal.stft(sliced_foreground, frame_length=
                model_settings['window_size_samples'], frame_step=
                model_settings['window_stride_samples'], fft_length=None,
                window_fn=tf.signal.hann_window)
            spectrograms = tf.abs(stfts)
            num_spectrogram_bins = stfts.shape[-1]
            lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40
            linear_to_mel_weight_matrix = (tf.signal.
                linear_to_mel_weight_matrix(num_mel_bins,
                num_spectrogram_bins, model_settings['sample_rate'],
                lower_edge_hertz, upper_edge_hertz))
            mel_spectrograms = tf.tensordot(spectrograms,
                linear_to_mel_weight_matrix, 1)
            mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]))
            log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-06)
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
                log_mel_spectrograms)[..., :model_settings[
                'dct_coefficient_count']]
            mfccs = tf.reshape(mfccs, [model_settings['spectrogram_length'],
                model_settings['dct_coefficient_count'], 1])
            next_element['audio'] = mfccs
        elif model_settings['feature_type'] == 'lfbe':
            preemphasis_coef = 1 - 2 ** -5
            power_offset = 52
            num_mel_bins = model_settings['dct_coefficient_count']
            paddings = tf.constant([[0, 0], [1, 0]])
            sliced_foreground = tf.expand_dims(sliced_foreground, 0)
            sliced_foreground = tf.pad(tensor=sliced_foreground, paddings=
                paddings, mode='CONSTANT')
            sliced_foreground = sliced_foreground[:, 1:
                ] - preemphasis_coef * sliced_foreground[:, :-1]
            sliced_foreground = tf.squeeze(sliced_foreground)
            stfts = tf.signal.stft(sliced_foreground, frame_length=
                model_settings['window_size_samples'], frame_step=
                model_settings['window_stride_samples'], fft_length=None,
                window_fn=functools.partial(tf.signal.hamming_window,
                periodic=False), pad_end=False, name='STFT')
            magspec = tf.abs(stfts)
            num_spectrogram_bins = magspec.shape[-1]
            powspec = 1 / model_settings['window_size_samples'] * tf.square(
                magspec)
            powspec_max = tf.reduce_max(input_tensor=powspec)
            powspec = tf.clip_by_value(powspec, 1e-30, powspec_max)

            def log10(x):
                numerator = tf.math.log(x)
                denominator = tf.math.log(tf.constant(10, dtype=numerator.
                    dtype))
                return numerator / denominator
            lower_edge_hertz, upper_edge_hertz = 0.0, model_settings[
                'sample_rate'] / 2.0
            linear_to_mel_weight_matrix = (tf.signal.
                linear_to_mel_weight_matrix(num_mel_bins=num_mel_bins,
                num_spectrogram_bins=num_spectrogram_bins, sample_rate=
                model_settings['sample_rate'], lower_edge_hertz=
                lower_edge_hertz, upper_edge_hertz=upper_edge_hertz))
            mel_spectrograms = tf.tensordot(powspec,
                linear_to_mel_weight_matrix, 1)
            mel_spectrograms.set_shape(magspec.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]))
            log_mel_spec = 10 * log10(mel_spectrograms)
            log_mel_spec = tf.expand_dims(log_mel_spec, -1, name='mel_spec')
            log_mel_spec = (log_mel_spec + power_offset - 32 + 32.0) / 64.0
            log_mel_spec = tf.clip_by_value(log_mel_spec, 0, 1)
            next_element['audio'] = log_mel_spec
        elif model_settings['feature_type'] == 'td_samples':
            paddings = [[0, 16000 - tf.shape(sliced_foreground)[0]]]
            wav_padded = tf.pad(sliced_foreground, paddings)
            wav_padded = tf.expand_dims(wav_padded, -1)
            wav_padded = tf.expand_dims(wav_padded, -1)
            next_element['audio'] = wav_padded
        return next_element
    return prepare_processing_graph
