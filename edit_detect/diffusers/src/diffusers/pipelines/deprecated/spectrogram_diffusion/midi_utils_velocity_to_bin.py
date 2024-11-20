def velocity_to_bin(velocity, num_velocity_bins):
    if velocity == 0:
        return 0
    else:
        return math.ceil(num_velocity_bins * velocity / note_seq.
            MAX_MIDI_VELOCITY)
