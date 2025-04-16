def set_device_states(devices, states) ->None:
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)
