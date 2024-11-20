def time_shifting(self, x):
    frame_num = len(x)
    shift_len = random.randint(0, frame_num - 1)
    new_sample = np.concatenate([x[shift_len:], x[:shift_len]], axis=0)
    return new_sample
