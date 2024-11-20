def representative_dataset_gen():
    for _ in range(num_calibration_steps):
        next_input = next(ds_iter)[0]
        yield [next_input]
