def fill_event_start_indices_to_cur_step():
    while len(event_start_indices) < len(frame_times) and frame_times[len(
        event_start_indices)] < cur_step / codec.steps_per_second:
        event_start_indices.append(cur_event_idx)
        state_event_indices.append(cur_state_event_idx)
