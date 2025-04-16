def encode_and_index_events(state, event_times, event_values, codec,
    frame_times, encode_event_fn, encoding_state_to_events_fn=None):
    """Encode a sequence of timed events and index to audio frame times.

    Encodes time shifts as repeated single step shifts for later run length encoding.

    Optionally, also encodes a sequence of "state events", keeping track of the current encoding state at each audio
    frame. This can be used e.g. to prepend events representing the current state to a targets segment.

    Args:
      state: Initial event encoding state.
      event_times: Sequence of event times.
      event_values: Sequence of event values.
      encode_event_fn: Function that transforms event value into a sequence of one
          or more Event objects.
      codec: An Codec object that maps Event objects to indices.
      frame_times: Time for every audio frame.
      encoding_state_to_events_fn: Function that transforms encoding state into a
          sequence of one or more Event objects.

    Returns:
      events: Encoded events and shifts. event_start_indices: Corresponding start event index for every audio frame.
          Note: one event can correspond to multiple audio indices due to sampling rate differences. This makes
          splitting sequences tricky because the same event can appear at the end of one sequence and the beginning of
          another.
      event_end_indices: Corresponding end event index for every audio frame. Used
          to ensure when slicing that one chunk ends where the next begins. Should always be true that
          event_end_indices[i] = event_start_indices[i + 1].
      state_events: Encoded "state" events representing the encoding state before
          each event.
      state_event_indices: Corresponding state event index for every audio frame.
    """
    indices = np.argsort(event_times, kind='stable')
    event_steps = [round(event_times[i] * codec.steps_per_second) for i in
        indices]
    event_values = [event_values[i] for i in indices]
    events = []
    state_events = []
    event_start_indices = []
    state_event_indices = []
    cur_step = 0
    cur_event_idx = 0
    cur_state_event_idx = 0

    def fill_event_start_indices_to_cur_step():
        while len(event_start_indices) < len(frame_times) and frame_times[len
            (event_start_indices)] < cur_step / codec.steps_per_second:
            event_start_indices.append(cur_event_idx)
            state_event_indices.append(cur_state_event_idx)
    for event_step, event_value in zip(event_steps, event_values):
        while event_step > cur_step:
            events.append(codec.encode_event(Event(type='shift', value=1)))
            cur_step += 1
            fill_event_start_indices_to_cur_step()
            cur_event_idx = len(events)
            cur_state_event_idx = len(state_events)
        if encoding_state_to_events_fn:
            for e in encoding_state_to_events_fn(state):
                state_events.append(codec.encode_event(e))
        for e in encode_event_fn(state, event_value, codec):
            events.append(codec.encode_event(e))
    while cur_step / codec.steps_per_second <= frame_times[-1]:
        events.append(codec.encode_event(Event(type='shift', value=1)))
        cur_step += 1
        fill_event_start_indices_to_cur_step()
        cur_event_idx = len(events)
    event_end_indices = event_start_indices[1:] + [len(events)]
    events = np.array(events).astype(np.int32)
    state_events = np.array(state_events).astype(np.int32)
    event_start_indices = segment(np.array(event_start_indices).astype(np.
        int32), TARGET_FEATURE_LENGTH)
    event_end_indices = segment(np.array(event_end_indices).astype(np.int32
        ), TARGET_FEATURE_LENGTH)
    state_event_indices = segment(np.array(state_event_indices).astype(np.
        int32), TARGET_FEATURE_LENGTH)
    outputs = []
    for start_indices, end_indices, event_indices in zip(event_start_indices,
        event_end_indices, state_event_indices):
        outputs.append({'inputs': events, 'event_start_indices':
            start_indices, 'event_end_indices': end_indices, 'state_events':
            state_events, 'state_event_indices': event_indices})
    return outputs
