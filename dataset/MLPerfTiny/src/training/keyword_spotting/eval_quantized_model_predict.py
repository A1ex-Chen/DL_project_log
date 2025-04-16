def predict(interpreter, data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_data = np.array(data, dtype=np.int8)
    output_data = np.empty_like(data)
    interpreter.set_tensor(input_details[0]['index'], input_data[i:i + 1, :])
    interpreter.invoke()
    output_data[i:i + 1, :] = interpreter.get_tensor(output_details[0]['index']
        )
    return output_data
