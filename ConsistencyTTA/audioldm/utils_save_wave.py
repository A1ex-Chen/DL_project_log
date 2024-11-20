def save_wave(waveform, savepath, name='outwav'):
    if type(name) is not list:
        name = [name] * waveform.shape[0]
    for i in range(waveform.shape[0]):
        path = os.path.join(savepath, '%s_%s.wav' % (os.path.basename(name[
            i]) if not '.wav' in name[i] else os.path.basename(name[i]).
            split('.')[0], i))
        print('Save audio to %s' % path)
        sf.write(path, waveform[i, 0], samplerate=16000)
