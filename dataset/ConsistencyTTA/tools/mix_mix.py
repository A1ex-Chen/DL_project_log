def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.0) * (1 - r) / r)
    sound = (sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2)
    return sound
