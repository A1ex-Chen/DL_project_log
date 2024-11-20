def prepare_background_data(bg_path, BACKGROUND_NOISE_DIR_NAME):
    """Searches a folder for background noise audio, and loads it into memory.
  It's expected that the background audio samples will be in a subdirectory
  named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
  the sample rate of the training data, but can be much longer in duration.
  If the '_background_noise_' folder doesn't exist at all, this isn't an
  error, it's just taken to mean that no background noise augmentation should
  be used. If the folder does exist, but it's empty, that's treated as an
  error.
  Returns:
    List of raw PCM-encoded audio samples of background noise.
  Raises:
    Exception: If files aren't found in the folder.
  """
    background_data = []
    background_dir = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME)
    if not os.path.exists(background_dir):
        return background_data
    search_path = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME, '*.wav')
    for wav_path in gfile.Glob(search_path):
        raw_audio = tf.io.read_file(wav_path)
        audio = tf.audio.decode_wav(raw_audio)
        background_data.append(audio[0])
    if not background_data:
        raise Exception('No background wav files were found in ' + search_path)
    return background_data
