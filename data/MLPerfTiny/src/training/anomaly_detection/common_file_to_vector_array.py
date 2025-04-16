def file_to_vector_array(file_name, n_mels=64, frames=5, n_fft=1024,
    hop_length=512, power=2.0, method='librosa', save_png=False, save_hist=
    False, save_bin=False, save_parts=False):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    dims = n_mels * frames
    y, sr = file_load(file_name)
    if method == 'librosa':
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=
            n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
        log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram +
            sys.float_info.epsilon)
    else:
        logger.error('spectrogram method not supported: {}'.format(method))
        return numpy.empty((0, dims))
    log_mel_spectrogram = log_mel_spectrogram[:, 50:250]
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1
    if vector_array_size < 1:
        return numpy.empty((0, dims))
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t:n_mels * (t + 1)] = log_mel_spectrogram[
            :, t:t + vector_array_size].T
    if save_png:
        save_path = file_name.replace('.wav', '_hist_' + method + '.png')
        librosa.display.specshow(log_mel_spectrogram)
        pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
        pylab.close()
    if save_hist:
        save_path = file_name.replace('.wav', '_hist_' + method + '.txt')
        numpy.swapaxes(log_mel_spectrogram, 0, 1).tofile(save_path, sep=',')
    if save_bin:
        save_path = file_name.replace('.wav', '_hist_' + method + '.bin')
        numpy.swapaxes(log_mel_spectrogram, 0, 1).astype('float32').tofile(
            save_path)
    if save_parts:
        for i in range(vector_array_size):
            save_path = file_name.replace('.wav', '_hist_' + method +
                '_part{0:03d}'.format(i) + '.bin')
            vector_array[i].astype('float32').tofile(save_path)
    return vector_array
